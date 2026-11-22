"""
promo_engine.py  -  Mini "PromoAI" simulation engine (no Streamlit code in here)

Inspired by the PepsiCo paper (Llenas et al.): demand model -> optimizer -> business constraints.
Scaled down on purpose:
    * 3 products, 3 weeks (21 days)
    * each product may run ONE promotion that lasts `window` days (user adjustable)
    * the optimizer is a real MILP (mixed-integer linear program), solved with HiGHS via
      scipy.optimize.milp - the same family of method PromoAI runs with Gurobi.
      A brute-force search is kept ONLY to cross-check the MILP and to draw the "landscape".

Everything is plain numpy/pandas so each step can be read and explained.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, kron, vstack

# --------------------------------------------------------------------------------------
# 1. DATA CLASSES
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Product:
    name: str
    base_units: float      # baseline units sold per day with no promotion
    shelf_price: float     # price consumers pay (retailer shelf price)
    wholesale: float       # price retailer pays PepsiCo (sell-in price)
    unit_cost: float       # PepsiCo cost per unit (COGS + distribution)
    elasticity: float      # promo price elasticity (negative): units *= (1-discount)^elasticity


@dataclass(frozen=True)
class Config:
    days: int = 21                      # 3 weeks
    window: int = 3                     # promotion length in days  <-- adjustable "x"
    depths: tuple = (0.10, 0.20, 0.30)  # discount depths the optimizer may pick from
    cannibalization: float = 0.5        # sales lost per unit of average discount on OTHER products
    dip: float = 0.20                   # post-promo "pantry loading" demand dip
    event_start: int = 12               # day (1-21) a demand event starts (e.g. game weekend)
    event_len: int = 3                  # event length in days
    event_boost: float = 0.30           # +30% baseline demand on event days
    funding_share: float = 0.4          # share of the shelf discount paid by PepsiCo (trade spend)
    w_pepsico: float = 0.6              # objective weight on PepsiCo profit (rest goes to retailer)
    max_concurrent: int = 2             # retailer slot capacity: max products on promo the same day
    min_margin_ratio: float = 0.30      # PepsiCo profit / PepsiCo revenue must stay >= this
    trade_budget: Optional[float] = None  # optional cap on total PepsiCo trade spend ($)
    mip_gap: float = 0.0                # MILP stopping gap (paper: 1-5% in production; 0 = prove optimal)
    time_limit: float = 60.0            # MILP time limit in seconds

    @property
    def sim_days(self) -> int:
        """Planning horizon + a tail of `window` days so the post-promo dip is never hidden
        by the end of the calendar (otherwise the optimizer would just promote on the last days)."""
        return self.days + self.window


def season(cfg: "Config") -> np.ndarray:
    """Demand multiplier per day: 1.0 normally, 1+boost on event days."""
    m = np.ones(cfg.sim_days)
    a = cfg.event_start - 1
    m[a:a + cfg.event_len] = 1 + cfg.event_boost
    return m


DEFAULT_PRODUCTS = (
    Product("Cola 12pk",   100, 2.00, 1.40, 0.90, -5.0),
    Product("Orange Soda",  60, 2.00, 1.40, 0.95, -4.5),
    Product("Chips",        80, 3.00, 2.20, 1.40, -4.0),
)

Option = tuple  # (start_day or None, discount depth).  (None, 0.0) means "no promotion"
NO_PROMO: Option = (None, 0.0)

# --------------------------------------------------------------------------------------
# 2. HELPERS: turn a promotion option into daily arrays
# --------------------------------------------------------------------------------------


def _daily_arrays(opt: Option, cfg: Config):
    """Return (discount[T], dip_flag[T]) for one product's promotion option."""
    disc = np.zeros(cfg.sim_days)
    dipf = np.zeros(cfg.sim_days)
    start, depth = opt
    if start is not None:
        disc[start:start + cfg.window] = depth                       # promo days
        dipf[start + cfg.window:start + 2 * cfg.window] = 1.0        # same-length dip afterwards
    return disc, dipf


def build_options(cfg: Config):
    """All promotion options for ONE product: 'none' + (every legal start day x every depth)."""
    opts = [NO_PROMO]
    for s in range(cfg.days - cfg.window + 1):      # promo must fit inside the 3 weeks
        for d in cfg.depths:
            opts.append((s, d))
    D = np.zeros((len(opts), cfg.sim_days))
    Z = np.zeros((len(opts), cfg.sim_days))
    for k, o in enumerate(opts):
        D[k], Z[k] = _daily_arrays(o, cfg)
    return opts, D, Z


# --------------------------------------------------------------------------------------
# 3. EVALUATE ONE SCHEDULE  (slow, explicit, easy to read -> used for explanations/tests)
# --------------------------------------------------------------------------------------


@dataclass
class Result:
    schedule: list
    kpis: dict
    daily: pd.DataFrame


def evaluate(products, cfg: Config, schedule) -> Result:
    """Simulate day by day. One 'schedule' = one promotion option per product."""
    P, T = len(products), cfg.sim_days
    season_t = season(cfg)
    disc = np.zeros((P, T))
    dipf = np.zeros((P, T))
    for p, opt in enumerate(schedule):
        disc[p], dipf[p] = _daily_arrays(opt, cfg)

    rows = []
    pep_profit = pep_rev = ret_profit = trade = units_tot = 0.0
    for t in range(T):
        for p, prod in enumerate(products):
            d = disc[p, t]
            others = (disc[:, t].sum() - d) / (P - 1)        # avg discount of competing own products
            lift = (1 - d) ** prod.elasticity                # price-response (>1 when discounted)
            units = (prod.base_units * season_t[t] * lift      # event days sell more
                     * (1 - cfg.dip * dipf[p, t])            # pantry-loading dip after a promo
                     * (1 - cfg.cannibalization * others))   # cannibalization by other promos
            trade_unit = cfg.funding_share * prod.shelf_price * d
            pep_unit_rev = prod.wholesale - trade_unit
            pep_unit_margin = pep_unit_rev - prod.unit_cost
            ret_unit_margin = (prod.shelf_price * (1 - d)) - pep_unit_rev
            pep_profit += units * pep_unit_margin
            pep_rev += units * pep_unit_rev
            ret_profit += units * ret_unit_margin
            trade += units * trade_unit
            units_tot += units
            rows.append(dict(day=t + 1, product=prod.name, discount=d, units=units,
                             tail=t >= cfg.days))

    daily = pd.DataFrame(rows)
    active = (disc > 0).sum(axis=0)
    kpis = dict(
        pepsico_profit=pep_profit, pepsico_revenue=pep_rev, retailer_profit=ret_profit,
        trade_spend=trade, units=units_tot,
        margin_ratio=pep_profit / pep_rev,
        max_concurrent=int(active.max()),
        objective=cfg.w_pepsico * pep_profit + (1 - cfg.w_pepsico) * ret_profit,
    )
    kpis["violations"] = check_constraints(kpis, cfg)
    kpis["feasible"] = len(kpis["violations"]) == 0
    return Result(list(schedule), kpis, daily)


def check_constraints(k: dict, cfg: Config) -> list:
    v = []
    if k["max_concurrent"] > cfg.max_concurrent:
        v.append(f"Slot capacity: {k['max_concurrent']} products on promo at once (max {cfg.max_concurrent})")
    if k["margin_ratio"] < cfg.min_margin_ratio - 1e-12:
        v.append(f"Margin floor: {k['margin_ratio']:.1%} < {cfg.min_margin_ratio:.1%}")
    if cfg.trade_budget is not None and k["trade_spend"] > cfg.trade_budget + 1e-9:
        v.append(f"Trade budget: ${k['trade_spend']:.0f} > ${cfg.trade_budget:.0f}")
    return v


# --------------------------------------------------------------------------------------
# 4. THE OPTIMIZER: MILP  (the PromoAI approach)
# --------------------------------------------------------------------------------------
#
#  Decision variables
#     x[p,o]  binary   1 if product p uses promotion option o      o = "none" or (start day s, depth d)
#     y[a,o,b,o'] in [0,1]   "pair" variable that equals x[a,o] * x[b,o']
#
#  Why y?  Cannibalization: product a's sales depend on product b's discount, so the profit
#  contains products of two binary choices (x[a,o]*x[b,o']).  A MILP must be LINEAR, so we replace
#  each product by y and add the linking constraints
#        sum_o' y[a,o,b,o'] = x[a,o]     and     sum_o y[a,o,b,o'] = x[b,o']
#  which force y = x*x' exactly (this is a standard "assignment linearization", and it is tight).
#
#  Model
#     maximise   w * PepsiCo profit + (1-w) * retailer profit          (linear in x and y)
#     s.t.       sum_o x[p,o] = 1                         one option per product (promo exclusivity)
#                sum_p sum_{o on promo day t} x[p,o] <= N  slot capacity, every day t
#                PepsiCo profit - gamma * PepsiCo revenue >= 0     margin floor
#                trade spend <= budget                      (optional)
#                linking constraints above;  x binary, 0 <= y <= 1
#  The promotion length x (days) and the price response (1-d)^elasticity are baked into each
#  option's daily numbers, so no piecewise-linear approximation is needed here. (The paper needs
#  PWL because it embeds a black-box LightGBM curve with a continuous discount variable.)


@dataclass
class OptResult:
    best: Optional[Result]
    n_total: int                                   # size of the search space (options^3)
    method: str = "milp"
    n_feasible: Optional[int] = None               # only known for brute force
    feasible_objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    info: dict = field(default_factory=dict)       # solver facts (status, time, nodes, model size...)


def _option_economics(products, cfg: Config, D, Z):
    """Per-product, per-option, per-day building blocks (they only depend on the product's OWN option)."""
    season_t = season(cfg)
    own = []
    for prod in products:
        U = prod.base_units * season_t[None, :] * (1 - D) ** prod.elasticity * (1 - cfg.dip * Z)
        trade = cfg.funding_share * prod.shelf_price * D
        rev = prod.wholesale - trade
        own.append(dict(U=U, trade=trade, rev=rev, pm=rev - prod.unit_cost,
                        rm=prod.shelf_price * (1 - D) - rev))
    return own


def optimize_milp(products, cfg: Config) -> OptResult:
    assert len(products) == 3, "prototype is built for exactly 3 products"
    P, T = 3, cfg.sim_days
    opts, D, Z = build_options(cfg)
    n = len(opts)
    own = _option_economics(products, cfg, D, Z)
    kap = cfg.cannibalization / (P - 1)
    pairs = [(0, 1), (0, 2), (1, 2)]
    nx, ny = P * n, n * n
    N = nx + len(pairs) * ny                       # total variables

    def metric(mu):
        """Turn a per-unit value mu[p][o,t] into linear coefficients on x and on y.
        total = sum_p sum_o (sum_t mu*U) x[p,o]  -  kappa/(P-1) * sum (sum_t mu_a*U_a*D_b) y[...]"""
        v = np.zeros(N)
        v[:nx] = np.array([(mu[p] * own[p]["U"]).sum(1) for p in range(P)]).ravel()
        for k, (a, b) in enumerate(pairs):
            a_lost_to_b = (mu[a] * own[a]["U"]) @ D.T        # [o of a, o' of b]
            b_lost_to_a = D @ (mu[b] * own[b]["U"]).T        # [o of a, o' of b]
            v[nx + k * ny: nx + (k + 1) * ny] = (-kap * (a_lost_to_b + b_lost_to_a)).ravel()
        return v

    def row(*blocks):                              # place sparse blocks side by side
        return hstack(blocks, format="csr")

    def z(r, c):
        return csr_matrix((r, c))

    A, lo, hi = [], [], []
    # (1) exactly one option per product
    for p in range(P):
        r = np.zeros(N); r[p * n:(p + 1) * n] = 1
        A.append(csr_matrix(r)); lo.append(1); hi.append(1)
    # (2) linking constraints  y = x * x'
    I, ones = identity(n, format="csr"), csr_matrix(np.ones((1, n)))
    for k, (a, b) in enumerate(pairs):
        for R, p in ((kron(I, ones), a), (kron(ones, I), b)):
            xblocks = [(-I if q == p else z(n, n)) for q in range(P)]
            yblocks = [(R if j == k else z(n, ny)) for j in range(len(pairs))]
            A.append(row(*xblocks, *yblocks)); lo += [0] * n; hi += [0] * n
    # (3) slot capacity: products on promo on day t <= N
    S = np.zeros((T, N))
    for p in range(P):
        S[:, p * n:(p + 1) * n] = (D > 0).T.astype(float)
    A.append(csr_matrix(S)); lo += [-np.inf] * T; hi += [cfg.max_concurrent] * T
    # (4) margin floor: PepsiCo profit - gamma * revenue >= 0
    A.append(csr_matrix(metric([own[p]["pm"] - cfg.min_margin_ratio * own[p]["rev"] for p in range(P)])))
    lo.append(0); hi.append(np.inf)
    # (5) optional trade budget
    if cfg.trade_budget is not None:
        A.append(csr_matrix(metric([own[p]["trade"] for p in range(P)])))
        lo.append(-np.inf); hi.append(cfg.trade_budget)
    A = vstack(A, format="csr")

    obj = metric([cfg.w_pepsico * own[p]["pm"] + (1 - cfg.w_pepsico) * own[p]["rm"] for p in range(P)])
    integrality = np.zeros(N); integrality[:nx] = 1            # only x is integer; y follows
    t0 = time.time()
    res = milp(-obj, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
               bounds=Bounds(np.zeros(N), np.ones(N)),
               options=dict(mip_rel_gap=cfg.mip_gap, time_limit=cfg.time_limit, disp=False))
    info = dict(status=res.message, solve_time=time.time() - t0,
                nodes=getattr(res, "mip_node_count", None), gap=getattr(res, "mip_gap", None),
                n_binary=nx, n_continuous=N - nx, n_constraints=A.shape[0],
                proven_optimal=(res.status == 0))
    best = None
    if res.x is not None:
        sel = [opts[int(np.argmax(res.x[p * n:(p + 1) * n]))] for p in range(P)]
        best = evaluate(products, cfg, sel)                    # re-simulate winner explicitly
        info["milp_objective"] = float(-res.fun)
    return OptResult(best, n ** 3, "milp", None, np.array([]), info)


def optimize(products, cfg: Config, method: str = "milp") -> OptResult:
    """Default optimizer = MILP. method='bruteforce' is only for validation."""
    return optimize_milp(products, cfg) if method == "milp" else optimize_bruteforce(products, cfg)


# --------------------------------------------------------------------------------------
# 5. CROSS-CHECK: brute-force search  (validation + "landscape" chart only)
# --------------------------------------------------------------------------------------


def optimize_bruteforce(products, cfg: Config) -> OptResult:
    """Try EVERY combination of (option for product1, option for product2, option for product3),
    drop those that break a constraint, keep the best. Fine for 3 products; impossible at real scale.
    Used here to (a) prove the MILP is right and (b) draw the distribution of all feasible calendars."""
    assert len(products) == 3, "prototype is built for exactly 3 products"
    P = 3
    opts, D, Z = build_options(cfg)
    n = len(opts)
    own = _option_economics(products, cfg, D, Z)

    def view(a, p, i):   # broadcast helper -> shape [n, n, T]
        if p == 0:
            return a[i][None, None, :]
        return a[:, None, :] if p == 1 else a[None, :, :]

    best_obj, best_idx = -np.inf, None
    n_feasible = 0
    feas_obj = []
    t0 = time.time()
    for i in range(n):                                      # loop over product-1 option, vectorise the rest
        dd = [view(D, p, i) for p in range(P)]
        sumD = dd[0] + dd[1] + dd[2]
        pep_profit = pep_rev = ret_profit = trade = 0.0
        for p in range(P):
            others = (sumD - dd[p]) / (P - 1)
            units = view(own[p]["U"], p, i) * (1 - cfg.cannibalization * others)
            pep_profit = pep_profit + (units * view(own[p]["pm"], p, i)).sum(-1)
            pep_rev = pep_rev + (units * view(own[p]["rev"], p, i)).sum(-1)
            ret_profit = ret_profit + (units * view(own[p]["rm"], p, i)).sum(-1)
            trade = trade + (units * view(own[p]["trade"], p, i)).sum(-1)
        conc = ((dd[0] > 0).astype(int) + (dd[1] > 0) + (dd[2] > 0)).max(-1)
        obj = cfg.w_pepsico * pep_profit + (1 - cfg.w_pepsico) * ret_profit

        ok = (conc <= cfg.max_concurrent) & (pep_profit >= cfg.min_margin_ratio * pep_rev - 1e-9)
        if cfg.trade_budget is not None:
            ok &= trade <= cfg.trade_budget + 1e-9
        if ok.any():
            n_feasible += int(ok.sum())
            feas_obj.append(obj[ok])
            masked = np.where(ok, obj, -np.inf)
            j, k = np.unravel_index(np.argmax(masked), masked.shape)
            if masked[j, k] > best_obj:
                best_obj, best_idx = masked[j, k], (i, j, k)

    best = None
    if best_idx is not None:
        best = evaluate(products, cfg, [opts[x] for x in best_idx])
    return OptResult(best, n ** 3, "bruteforce", n_feasible,
                     np.concatenate(feas_obj) if feas_obj else np.array([]),
                     dict(solve_time=time.time() - t0))


# --------------------------------------------------------------------------------------
# 6. TEST CASES: manual "rule-of-thumb" calendars vs the optimizer
# --------------------------------------------------------------------------------------


def heuristic_schedules(cfg: Config) -> dict:
    """What a planner might do by hand. All evaluated under the SAME constraints."""
    mid = cfg.depths[len(cfg.depths) // 2]
    lo, hi = min(cfg.depths), max(cfg.depths)
    last = cfg.days - cfg.window
    wk = [min(0, last), min(7, last), min(14, last)]
    return {
        "A. No promotion (baseline)":            [NO_PROMO] * 3,
        "B. Everyone promotes together (day 1)": [(0, mid)] * 3,
        "C. One product per week, mid discount": [(s, mid) for s in wk],
        "D. One product per week, deep discount": [(s, hi) for s in wk],
        "E. One product per week, light discount": [(s, lo) for s in wk],
        "F. Back-to-back at start, mid discount": [(min(i * cfg.window, last), mid) for i in range(3)],
    }


def compare_scenarios(products, cfg: Config):
    """Evaluate every heuristic + the optimizer under the same config. Returns (DataFrame, OptResult)."""
    opt = optimize(products, cfg)
    cases = heuristic_schedules(cfg)
    results = {name: evaluate(products, cfg, s) for name, s in cases.items()}
    if opt.best is not None:
        results["G. OPTIMIZER (MILP)"] = opt.best
    rows = []
    for name, r in results.items():
        k = r.kpis
        rows.append({
            "Scenario": name, "Objective": k["objective"], "PepsiCo profit": k["pepsico_profit"],
            "Retailer profit": k["retailer_profit"], "Trade spend": k["trade_spend"],
            "Units": k["units"], "Margin %": k["margin_ratio"] * 100,
            "Max promos/day": k["max_concurrent"], "Feasible": k["feasible"],
            "Why infeasible": "; ".join(k["violations"]),
        })
    df = pd.DataFrame(rows)
    feas_best = df.loc[df["Feasible"], "Objective"].max() if df["Feasible"].any() else np.nan
    df["Best feasible?"] = df["Feasible"] & np.isclose(df["Objective"], feas_best)
    return df, opt, results


def constraint_whatifs(products, cfg: Config) -> pd.DataFrame:
    """Re-optimise under different business rules to show what each rule costs."""
    variants = {
        "Your current settings": cfg,
        "No slot limit (3 at once)": replace(cfg, max_concurrent=3),
        "Strict slots (1 at a time)": replace(cfg, max_concurrent=1),
        "No margin floor": replace(cfg, min_margin_ratio=0.0),
        "Tight margin floor (32%)": replace(cfg, min_margin_ratio=0.32),
        "PepsiCo-only objective": replace(cfg, w_pepsico=1.0),
        "Retailer-friendly objective (30/70)": replace(cfg, w_pepsico=0.3),
    }
    rows = []
    for name, c in variants.items():
        o = optimize(products, c)
        if o.best is None:
            rows.append({"Rule set": name, "Schedule": "INFEASIBLE"})
            continue
        k = o.best.kpis
        rows.append({"Rule set": name, "Solve time (s)": o.info["solve_time"],
                     "PepsiCo profit": k["pepsico_profit"], "Retailer profit": k["retailer_profit"],
                     "Trade spend": k["trade_spend"], "Margin %": k["margin_ratio"] * 100,
                     "Schedule": describe(products, o.best.schedule)})
    return pd.DataFrame(rows)


def window_sweep(products, cfg: Config, windows=range(1, 8)) -> pd.DataFrame:
    """Re-optimise for each promotion length x = 1..7 days."""
    rows = []
    for x in windows:
        c = replace(cfg, window=x)
        base = evaluate(products, c, [NO_PROMO] * 3).kpis      # same evaluation length for a fair gain
        o = optimize(products, c)
        if o.best is None:
            rows.append({"Window (days)": x, "Objective": np.nan, "Schedule": "INFEASIBLE"})
            continue
        k = o.best.kpis
        rows.append({"Window (days)": x, "Objective": k["objective"],
                     "Gain vs no promo ($)": k["objective"] - base["objective"],
                     "Gain %": (k["objective"] / base["objective"] - 1) * 100,
                     "PepsiCo profit": k["pepsico_profit"], "Retailer profit": k["retailer_profit"],
                     "Units": k["units"], "Schedule": describe(products, o.best.schedule)})
    return pd.DataFrame(rows)


def describe(products, schedule) -> str:
    parts = []
    for prod, (s, d) in zip(products, schedule):
        parts.append(f"{prod.name}: " + ("none" if s is None else f"day {s + 1} @ {d:.0%}"))
    return " | ".join(parts)


# --------------------------------------------------------------------------------------
# 7. SELF-CHECKS (used by pytest AND by the Streamlit "Test cases" tab)
# --------------------------------------------------------------------------------------


def run_self_checks(products, cfg: Config) -> list:
    """Returns list of (test name, passed, detail)."""
    cfg = replace(cfg, mip_gap=0.0)          # checks need proven-optimal MILP solutions
    out = []

    def add(name, passed, detail):
        out.append((name, bool(passed), detail))

    # T1 baseline sanity
    base = evaluate(products, cfg, [NO_PROMO] * 3)
    expected = sum(p.base_units for p in products) * season(cfg).sum()
    add("T1  No promotion => units equal baseline demand",
        np.isclose(base.kpis["units"], expected), f"{base.kpis['units']:.1f} vs {expected:.1f}")

    opt = optimize(products, cfg)
    if opt.best is None:
        add("T2  MILP finds a feasible schedule", False, "no feasible schedule - relax constraints")
        return out
    add("T2  MILP finds a feasible, proven-optimal schedule", opt.info["proven_optimal"],
        f"{opt.info['status']} in {opt.info['solve_time']:.1f}s")

    # T3 fast vectorised vs slow explicit simulation
    add("T3  Winner respects every constraint (checked by the day-by-day simulator)", opt.best.kpis["feasible"], "no violations")

    # T4 window length respected
    d = opt.best.daily
    lens = [(d[d["product"] == p.name]["discount"] > 0).sum() for p in products]
    add("T4  Every promotion lasts exactly x days (or is not run)",
        all(l in (0, cfg.window) for l in lens), f"promo days per product = {[int(x) for x in lens]}, x = {cfg.window}")

    # T5 optimizer beats every manual calendar
    _, _, res = compare_scenarios(products, cfg)
    manual = [r.kpis["objective"] for n, r in res.items() if not n.startswith("G.") and r.kpis["feasible"]]
    add("T5  MILP >= every feasible manual calendar",
        all(opt.best.kpis["objective"] >= m - 1e-9 for m in manual),
        f"optimizer {opt.best.kpis['objective']:.1f} vs best manual {max(manual):.1f}")

    # T6 relaxing constraints can never make the optimum worse
    relaxed = optimize(products, replace(cfg, max_concurrent=3, min_margin_ratio=0.0, trade_budget=None))
    add("T6  Relaxing constraints never lowers the optimum",
        relaxed.best.kpis["objective"] >= opt.best.kpis["objective"] - 1e-9,
        f"relaxed {relaxed.best.kpis['objective']:.1f} >= constrained {opt.best.kpis['objective']:.1f}")

    # T7 cannibalization really hurts simultaneous promos
    sched = [(0, 0.2)] * 3
    a = evaluate(products, replace(cfg, cannibalization=0.0), sched).kpis["units"]
    b = evaluate(products, cfg, sched).kpis["units"]
    add("T7  Cannibalization lowers units when promos overlap",
        a >= b - 1e-9, f"no cannibalization {a:.0f} units >= with {b:.0f} units")

    # T8 independent exhaustive search with the slow evaluator (tiny search space)
    tiny = replace(cfg, depths=(0.2,), window=max(cfg.window, 5))
    o_fast = optimize(products, tiny)
    opts, _, _ = build_options(tiny)
    best_slow = -np.inf
    for a_ in opts:
        for b_ in opts:
            for c_ in opts:
                r = evaluate(products, tiny, [a_, b_, c_])
                if r.kpis["feasible"] and r.kpis["objective"] > best_slow:
                    best_slow = r.kpis["objective"]
    add("T8  MILP == independent slow exhaustive search (tiny case)",
        o_fast.best is not None and np.isclose(o_fast.best.kpis["objective"], best_slow),
        f"{o_fast.best.kpis['objective']:.4f} vs {best_slow:.4f}")

    # T9 MILP vs vectorised brute force on the FULL search space
    bf = optimize_bruteforce(products, cfg)
    add("T9  MILP == brute-force optimum on the full search space",
        np.isclose(bf.best.kpis["objective"], opt.best.kpis["objective"], atol=1e-6),
        f"MILP {opt.best.kpis['objective']:.4f} vs brute force {bf.best.kpis['objective']:.4f} "
        f"({bf.n_total:,} calendars checked)")
    return out
