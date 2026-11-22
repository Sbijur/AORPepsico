"""
promo_tests.py  -  round-2 test catalogue for the Mini PromoAI engine.

Shared by the Streamlit app (tab "Automated tests") and by pytest (test_promo.py).
Every test uses the wiki base case (DEFAULT_PRODUCTS, Config()) so its expected value can be checked by hand.

Groups
  F  Formula      hand-checkable numbers from the demand and money model (wiki 4.4)
  M  Metamorphic  change an input in a known way, the answer must move predictably
  B  Boundary     elasticity range -2 to -4 where promotions switch on
  E  Edge         odd inputs, solver limits, error handling
Each test returns a short "observed" string and raises AssertionError on failure.
"""
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from promo_engine import (DEFAULT_PRODUCTS, NO_PROMO, Config, build_options, describe, evaluate,
                          optimize, optimize_bruteforce)

BASE = Config()                 # wiki base case: x=3, depths 10/20/30%, slot 2, floor 30%
P0 = DEFAULT_PRODUCTS
_STATE = {}                     # shares the elasticity switch point found by B01 with B02


def units_on(res, product, day):
    d = res.daily
    return float(d[(d["product"] == product) & (d["day"] == day)]["units"].iloc[0])


def opt_obj(products, cfg):
    o = optimize(products, cfg)
    return None if o.best is None else o.best.kpis["objective"]


# ---------------------------------------------------------------- F: formula tests (hand-checkable)
def f01():
    # Cola 30% off on days 1-3, nobody else promoting, no event, no dip yet:
    # units = 100 * (1-0.3)^-5 = 594.99
    r = evaluate(P0, BASE, [(0, 0.30), NO_PROMO, NO_PROMO])
    u = units_on(r, "Cola 12pk", 1)
    exp = 100 * 0.7 ** -5
    assert abs(u - exp) < 1e-6, f"{u:.2f} != {exp:.2f}"
    return f"{u:.2f} units (= 100 x 0.7^-5)"

def f02():
    # Unit money at 30% off for Cola (wiki 4.4 worked example): trade 0.24, PepsiCo margin 0.26 (was 0.50).
    # Check the SIMULATOR against a hand calculation of the change in PepsiCo profit when Cola runs
    # 30% off on days 1-3 (no event; others not promoting):
    #   Cola promo days : 3 x (594.99 x 0.26 - 100 x 0.50)
    #   Cola dip days   : 3 x (80 x 0.50 - 100 x 0.50)
    #   cannibalisation : Orange and Chips see avg rival discount 0.15 -> lose 7.5% of units on days 1-3
    #                     3 x (60 x 0.075 x 0.45 + 80 x 0.075 x 0.80)
    d, p = 0.30, P0[0]
    trade = BASE.funding_share * p.shelf_price * d
    pep_m = p.wholesale - trade - p.unit_cost
    ret_m = p.shelf_price * (1 - d) - (p.wholesale - trade)
    assert abs(trade - 0.24) < 1e-9 and abs(pep_m - 0.26) < 1e-9 and abs(ret_m - 0.24) < 1e-9
    hand = 3 * (100 * 0.7 ** -5 * 0.26 - 100 * 0.50) + 3 * (80 * 0.50 - 100 * 0.50) \
        - 3 * (60 * 0.075 * 0.45 + 80 * 0.075 * 0.80)
    sim = (evaluate(P0, BASE, [(0, 0.30), NO_PROMO, NO_PROMO]).kpis["pepsico_profit"]
           - evaluate(P0, BASE, [NO_PROMO] * 3).kpis["pepsico_profit"])
    assert abs(sim - hand) < 1e-6, (sim, hand)
    return f"unit trade 0.24 / margins 0.26 & 0.24; PepsiCo profit change {sim:+.2f} = hand calc {hand:+.2f}"

def f03():
    # Post-promo dip: promo days 1-3 -> days 4-6 at 80% of base, day 7 back to base
    r = evaluate(P0, BASE, [(0, 0.30), NO_PROMO, NO_PROMO])
    dip = [units_on(r, "Cola 12pk", t) for t in (4, 5, 6)]
    back = units_on(r, "Cola 12pk", 7)
    assert all(abs(u - 80) < 1e-9 for u in dip) and abs(back - 100) < 1e-9, (dip, back)
    return f"days 4-6 = {dip[0]:.0f}, day 7 = {back:.0f}"

def f04():
    # Event boost: with no promotion, days 12-14 sell 1.3x base; day 11 and 15 do not
    r = evaluate(P0, BASE, [NO_PROMO] * 3)
    ev = [units_on(r, "Orange Soda", t) for t in (11, 12, 13, 14, 15)]
    assert ev == [60, 78, 78, 78, 60] or np.allclose(ev, [60, 78, 78, 78, 60]), ev
    return f"Orange Soda days 11-15 = {[round(x) for x in ev]}"

def f05():
    # Cannibalisation: Orange & Chips at 20% on days 1-3, Cola not promoting:
    # Cola units = 100 * (1 - 0.5 * 0.2) = 90
    r = evaluate(P0, BASE, [NO_PROMO, (0, 0.20), (0, 0.20)])
    u = units_on(r, "Cola 12pk", 1)
    assert abs(u - 90) < 1e-9, u
    return f"Cola day 1 = {u:.1f} (= 100 x (1 - 0.5 x 0.2))"

def f06():
    # Accounting identity per unit: PepsiCo margin + retailer margin + unit cost = shelf price paid
    # => PepsiCo profit + retailer profit + sum(cost x units) = consumer spend
    sched = [(11, 0.30), (13, 0.20), (14, 0.10)]
    r = evaluate(P0, BASE, sched)
    d = r.daily.merge(pd.DataFrame([vars(p) for p in P0]).rename(columns={"name": "product"}), on="product")
    spend = (d.units * d.shelf_price * (1 - d.discount)).sum()
    cost = (d.units * d.unit_cost).sum()
    lhs = r.kpis["pepsico_profit"] + r.kpis["retailer_profit"] + cost
    assert abs(lhs - spend) < 1e-6, (lhs, spend)
    return f"{lhs:,.2f} = consumer spend {spend:,.2f}"

def f07():
    # Option count: 1 + (21 - x + 1) x |depths|  ->  x=3: 58, x=7: 46; search space = n^3
    n3 = len(build_options(BASE)[0]); n7 = len(build_options(replace(BASE, window=7))[0])
    o = optimize(P0, BASE)
    assert n3 == 58 and n7 == 46 and o.n_total == 58 ** 3 == 195_112, (n3, n7, o.n_total)
    return f"x=3: {n3} options, 195,112 calendars; x=7: {n7} options"

def f08():
    # Simulated horizon = 21 + x tail days, and a promo on the last legal day has its dip in the tail
    c = BASE
    last = c.days - c.window                       # 0-based start of the last legal promo (day 19)
    r = evaluate(P0, c, [(last, 0.30), NO_PROMO, NO_PROMO])
    tail = r.daily[(r.daily["product"] == "Cola 12pk") & r.daily["tail"]]
    assert c.sim_days == 24 and len(tail) == 3 and np.allclose(tail.units, 80), tail.units.tolist()
    return "24 simulated days; dip of a day-19 promo lands on tail days 22-24 (80 units each)"

def f09():
    # Funding share 0: PepsiCo bears no discount -> trade spend 0 and PepsiCo margin/unit unchanged by promos
    c = replace(BASE, funding_share=0.0)
    r = evaluate(P0, c, [(0, 0.30), NO_PROMO, NO_PROMO])
    assert r.kpis["trade_spend"] == 0
    return "trade spend 0; retailer absorbs the full discount"


# ---------------------------------------------------------------- M: metamorphic tests
def m01():
    # Product order must not matter: permute products -> same objective and same per-product plan
    o1 = optimize(P0, BASE)
    perm = (P0[2], P0[0], P0[1])
    o2 = optimize(perm, BASE)
    a = dict(zip([p.name for p in P0], o1.best.schedule))
    b = dict(zip([p.name for p in perm], o2.best.schedule))
    assert abs(o1.best.kpis["objective"] - o2.best.kpis["objective"]) < 1e-6 and a == b, (a, b)
    return f"objective {o1.best.kpis['objective']:.4f} both ways, same plan"

def m02():
    # Doubling every product's base volume doubles the objective and keeps the plan
    # (every term is linear in units; the margin-floor ratio is unchanged)
    big = tuple(replace(p, base_units=2 * p.base_units) for p in P0)
    o1, o2 = optimize(P0, BASE), optimize(big, BASE)
    assert abs(o2.best.kpis["objective"] - 2 * o1.best.kpis["objective"]) < 1e-6
    assert o1.best.schedule == o2.best.schedule
    return f"{o2.best.kpis['objective']:.2f} = 2 x {o1.best.kpis['objective']:.2f}, same plan"

def m03():
    # Scaling every price and cost by 1.5 (currency change) scales the objective by 1.5 and keeps the plan
    k = 1.5
    sc = tuple(replace(p, shelf_price=k * p.shelf_price, wholesale=k * p.wholesale, unit_cost=k * p.unit_cost) for p in P0)
    o1, o2 = optimize(P0, BASE), optimize(sc, BASE)
    assert abs(o2.best.kpis["objective"] - k * o1.best.kpis["objective"]) < 1e-6 and o1.best.schedule == o2.best.schedule
    return f"{o2.best.kpis['objective']:.2f} = 1.5 x {o1.best.kpis['objective']:.2f}, same plan"

def m04():
    # Moving the demand event to days 5-7 moves Cola's promotion with it
    c = replace(BASE, event_start=5)
    o = optimize(P0, c)
    cola_start = o.best.schedule[0][0]
    assert cola_start == 4, describe(P0, o.best.schedule)
    return describe(P0, o.best.schedule)

def m05():
    # Removing the event (boost 0) can only lower the optimum
    o_ev, o_no = opt_obj(P0, BASE), opt_obj(P0, replace(BASE, event_boost=0.0))
    assert o_no <= o_ev + 1e-9, (o_no, o_ev)
    return f"no event {o_no:,.1f} <= with event {o_ev:,.1f}"

def m06():
    # Adding a depth option (40%) never lowers the optimum; removing options never raises it
    full = opt_obj(P0, BASE)
    more = opt_obj(P0, replace(BASE, depths=(0.10, 0.20, 0.30, 0.40)))
    only20 = optimize(P0, replace(BASE, depths=(0.20,)))
    assert more >= full - 1e-9 >= only20.best.kpis["objective"] - 1e-9
    assert all(s is None or d == 0.20 for s, d in only20.best.schedule)
    return f"with 40%: {more:,.1f} >= base {full:,.1f} >= 20%-only {only20.best.kpis['objective']:,.1f}"

def m07():
    # Trade budget: looser budget never lowers the optimum, and spend always respects the budget
    rows = []
    for b in (0.0, 300.0, 600.0, 900.0, None):
        o = optimize(P0, replace(BASE, trade_budget=b))
        k = o.best.kpis
        assert b is None or k["trade_spend"] <= b + 1e-6, (b, k["trade_spend"])
        rows.append(k["objective"])
    assert all(rows[i] <= rows[i + 1] + 1e-9 for i in range(len(rows) - 1)), rows
    return "objective " + " <= ".join(f"{v:,.0f}" for v in rows) + " (budget 0/300/600/900/none)"

def m08():
    # More cannibalisation (0 -> 0.5 -> 1.0) never raises the optimum
    v = [opt_obj(P0, replace(BASE, cannibalization=c)) for c in (0.0, 0.5, 1.0)]
    assert v[0] >= v[1] - 1e-9 >= v[2] - 2e-9, v
    return " >= ".join(f"{x:,.1f}" for x in v)

def m09():
    # Deeper post-promo dip (0 -> 0.2 -> 0.5) never raises the optimum
    v = [opt_obj(P0, replace(BASE, dip=d)) for d in (0.0, 0.2, 0.5)]
    assert v[0] >= v[1] - 1e-9 >= v[2] - 2e-9, v
    return " >= ".join(f"{x:,.1f}" for x in v)

def m10():
    # Objective weight extremes: w=1 plan maximises PepsiCo profit, w=0 plan maximises retailer profit
    # -> each extreme must beat the base-weight plan on its own measure (brute-force verified)
    base = optimize(P0, BASE).best.kpis
    pep = optimize_bruteforce(P0, replace(BASE, w_pepsico=1.0)).best.kpis
    pep_m = optimize(P0, replace(BASE, w_pepsico=1.0)).best.kpis
    ret = optimize(P0, replace(BASE, w_pepsico=0.0)).best.kpis
    assert abs(pep_m["pepsico_profit"] - pep["pepsico_profit"]) < 1e-6
    assert pep_m["pepsico_profit"] >= base["pepsico_profit"] - 1e-9
    assert ret["retailer_profit"] >= base["retailer_profit"] - 1e-9
    return (f"w=1: PepsiCo {pep_m['pepsico_profit']:,.1f} (= brute force) >= {base['pepsico_profit']:,.1f}; "
            f"w=0: retailer {ret['retailer_profit']:,.1f} >= {base['retailer_profit']:,.1f}")

def m11():
    # The margin floor really binds in the base case (wiki: 30.002%)
    k = optimize(P0, BASE).best.kpis
    assert BASE.min_margin_ratio <= k["margin_ratio"] < BASE.min_margin_ratio + 0.001, k["margin_ratio"]
    return f"margin ratio {k['margin_ratio']:.4%} vs floor 30%"


# ---------------------------------------------------------------- B: elasticity boundary (wiki gap)
def b01():
    # Sweep a common elasticity from -2.0 to -4.0. Below some switch point the optimum is "no promotion";
    # past it the optimizer promotes. At every point the MILP must equal brute force (hardest region for errors).
    rows, switch = [], None
    for e in np.round(np.arange(-2.0, -4.01, -0.25), 2):
        prods = tuple(replace(p, elasticity=float(e)) for p in P0)
        m, bf = optimize(prods, BASE), optimize_bruteforce(prods, BASE)
        assert m.info["proven_optimal"]
        assert abs(m.best.kpis["objective"] - bf.best.kpis["objective"]) < 1e-6, (e, m.best.kpis["objective"], bf.best.kpis["objective"])
        n_promo = sum(s is not None for s, _ in m.best.schedule)
        if n_promo and switch is None:
            switch = e
        rows.append(f"{e}:{n_promo}")
    assert switch is not None
    _STATE["switch"] = switch
    return f"MILP = brute force at all 9 points; first promotion at elasticity {switch} (e:promos {', '.join(rows)})"

def b02():
    # Just past the switch point, the gain over no promotion must be small but positive,
    # and just before it the optimizer must return exactly the baseline
    s = _STATE.get("switch", -3.0)
    before = tuple(replace(p, elasticity=float(s + 0.25)) for p in P0)
    after = tuple(replace(p, elasticity=float(s)) for p in P0)
    ob, oa = optimize(before, BASE).best.kpis, optimize(after, BASE).best.kpis
    bb = evaluate(before, BASE, [NO_PROMO] * 3).kpis["objective"]
    ba = evaluate(after, BASE, [NO_PROMO] * 3).kpis["objective"]
    assert abs(ob["objective"] - bb) < 1e-6 and oa["objective"] > ba
    return f"at {s + 0.25}: = baseline {bb:,.1f}; at {s}: gain {oa['objective'] - ba:+,.1f} ({(oa['objective'] / ba - 1):+.2%})"

def b03():
    # Mixed elasticities: only the responsive product should promote (Cola -5, others -1.5)
    prods = (P0[0], replace(P0[1], elasticity=-1.5), replace(P0[2], elasticity=-1.5))
    o = optimize(prods, BASE)
    s = o.best.schedule
    assert s[0][0] is not None and s[1][0] is None and s[2][0] is None, describe(prods, s)
    return describe(prods, s)


# ---------------------------------------------------------------- E: edge cases & robustness
def e01():
    # Fewer or more than 3 products is not supported: must fail loudly (not silently give a wrong answer)
    msgs = []
    for prods in (P0[:1], P0[:2], P0 + (replace(P0[0], name="Cola 2L"),)):
        try:
            optimize(prods, BASE)
            raise AssertionError(f"{len(prods)} products accepted silently")
        except AssertionError as e:
            if "accepted silently" in str(e):
                raise
            msgs.append(len(prods))
    return f"clear error for {msgs} products ('prototype is built for exactly 3 products')"

def e02():
    # Promotion as long as the whole horizon (x = 21): only start day 1 is legal; solver still works
    c = replace(BASE, window=21)
    opts = build_options(c)[0]
    o = optimize(P0, c)
    assert len(opts) == 1 + 1 * 3 and o.best is not None and o.best.kpis["feasible"]
    return f"{len(opts)} options per product; plan: {describe(P0, o.best.schedule)}"

def e03():
    # Promotion longer than the horizon (x = 22): no promotion can fit -> optimizer returns the baseline
    c = replace(BASE, window=22)
    o = optimize(P0, c)
    base = evaluate(P0, c, [NO_PROMO] * 3).kpis["objective"]
    assert len(build_options(c)[0]) == 1 and abs(o.best.kpis["objective"] - base) < 1e-6
    return "only 'no promotion' exists; optimizer returns the baseline"

def e04():
    # Slot capacity 0 = promotions forbidden -> baseline; capacity 3 = slot rule irrelevant
    o0 = optimize(P0, replace(BASE, max_concurrent=0)).best
    assert all(s is None for s, _ in o0.schedule)
    return "capacity 0 -> no promotions"

def e05():
    # Tiny time limit must not crash: returns a (possibly sub-optimal) feasible plan or None
    o = optimize(P0, replace(BASE, time_limit=0.01))
    ok = o.best is None or o.best.kpis["feasible"]
    assert ok
    return f"status '{o.info['status'][:50]}', plan {'none' if o.best is None else 'feasible'}"

def e06():
    # 1% gap: result within 1% of the proven optimum (paper's high-value-market setting)
    opt = opt_obj(P0, BASE)
    g1 = optimize(P0, replace(BASE, mip_gap=0.01))
    assert g1.best.kpis["objective"] >= 0.99 * opt - 1e-9
    return f"{g1.best.kpis['objective']:,.1f} = {g1.best.kpis['objective'] / opt:.2%} of optimum"

def e07():
    # Determinism: two runs give exactly the same plan and objective
    a, b = optimize(P0, BASE), optimize(P0, BASE)
    assert a.best.schedule == b.best.schedule and a.best.kpis["objective"] == b.best.kpis["objective"]
    return "identical plan and objective"

def e08():
    # Each violation type is reported with its own message (constraint report, Req 4.1)
    c = replace(BASE, trade_budget=100.0, min_margin_ratio=0.34)
    r = evaluate(P0, c, [(0, 0.30)] * 3)
    v = " | ".join(r.kpis["violations"])
    assert "Slot capacity" in v and "Margin floor" in v and "Trade budget" in v, v
    return "slot, margin and budget violations all named"

def e09():
    # A product that loses money on every unit (wholesale < cost) must never be promoted
    loser = (P0[0], P0[1], replace(P0[2], wholesale=1.30))     # Chips cost 1.40 > sell-in 1.30
    o = optimize(loser, replace(BASE, min_margin_ratio=-1.0, w_pepsico=1.0))
    assert o.best.schedule[2][0] is None, describe(loser, o.best.schedule)
    return describe(loser, o.best.schedule)

def e10():
    # Zero-volume product: promoting it changes nothing for itself; plan must still be feasible
    z = (P0[0], P0[1], replace(P0[2], base_units=0.0))
    o = optimize(z, BASE)
    assert o.best is not None and o.best.kpis["feasible"]
    return describe(z, o.best.schedule)


TESTS = [
    ("F01", "Formula", "Promo lift: Cola 30% off sells 100 x 0.7^-5 units", "594.99 units", f01),
    ("F02", "Formula", "Unit money at 30% off and its effect on PepsiCo profit (wiki 4.4)", "simulator = hand calculation", f02),
    ("F03", "Formula", "Post-promotion dip lasts x days at 80%", "days 4-6 = 80, day 7 = 100", f03),
    ("F04", "Formula", "Demand event lifts days 12-14 by 30%", "60, 78, 78, 78, 60", f04),
    ("F05", "Formula", "Cannibalisation from two rivals at 20%", "Cola = 90 units", f05),
    ("F06", "Formula", "Money identity: profits + cost = consumer spend", "exact equality", f06),
    ("F07", "Formula", "Option count and search space", "58 options, 195,112 calendars", f07),
    ("F08", "Formula", "Tail days catch the dip of a late promotion", "dip on days 22-24", f08),
    ("F09", "Formula", "Funding share 0 -> no trade spend", "trade spend 0", f09),
    ("M01", "Metamorphic", "Re-ordering products changes nothing", "same objective and plan", m01),
    ("M02", "Metamorphic", "Doubling volumes doubles the objective", "2x objective, same plan", m02),
    ("M03", "Metamorphic", "Currency change (all prices x1.5)", "1.5x objective, same plan", m03),
    ("M04", "Metamorphic", "Event moved to days 5-7", "Cola starts day 5", m04),
    ("M05", "Metamorphic", "No demand event", "objective not higher", m05),
    ("M06", "Metamorphic", "More discount options / fewer options", "more >= base >= 20%-only", m06),
    ("M07", "Metamorphic", "Trade budget 0 -> 300 -> 600 -> 900 -> none", "non-decreasing, spend <= budget", m07),
    ("M08", "Metamorphic", "Cannibalisation 0 -> 0.5 -> 1.0", "non-increasing objective", m08),
    ("M09", "Metamorphic", "Post-promo dip 0 -> 20% -> 50%", "non-increasing objective", m09),
    ("M10", "Metamorphic", "Objective weight extremes w=1 / w=0", "each maximises its own party", m10),
    ("M11", "Metamorphic", "Margin floor binds in the base case", "ratio within 0.1 pt of 30%", m11),
    ("B01", "Boundary", "Elasticity sweep -2.0 to -4.0 (MILP vs brute force)", "equal at every point; find switch", b01),
    ("B02", "Boundary", "Either side of the switch point", "baseline before, small gain after", b02),
    ("B03", "Boundary", "Only the responsive product promotes", "Cola only", b03),
    ("E01", "Edge", "1, 2 or 4 products", "clear error, no silent answer", e01),
    ("E02", "Edge", "Promotion as long as the horizon (x = 21)", "4 options, feasible plan", e02),
    ("E03", "Edge", "Promotion longer than the horizon (x = 22)", "baseline returned", e03),
    ("E04", "Edge", "Slot capacity 0", "no promotions", e04),
    ("E05", "Edge", "Time limit 0.01 s", "no crash", e05),
    ("E06", "Edge", "1% gap tolerance", ">= 99% of optimum", e06),
    ("E07", "Edge", "Determinism", "identical results", e07),
    ("E08", "Edge", "Every violation type is named", "slot, margin, budget", e08),
    ("E09", "Edge", "Loss-making product (sell-in < cost)", "never promoted", e09),
    ("E10", "Edge", "Zero-volume product", "feasible plan", e10),
]


GROUPS = {"F": "Formula", "M": "Metamorphic", "B": "Boundary", "E": "Edge"}


def run_test(test):
    """Run one catalogue entry -> result dict."""
    tid, cat, desc, expected, fn = test
    t0 = time.time()
    try:
        obs, res = fn(), "PASS"
    except AssertionError as e:
        obs, res = f"assertion failed: {e}", "FAIL"
    except Exception as e:                                    # noqa: BLE001
        obs, res = f"error {type(e).__name__}: {e}", "FAIL"
    return dict(ID=tid, Group=cat, Test=desc, Expected=expected, Observed=obs, Result=res,
                Seconds=round(time.time() - t0, 2))


def run_round2(ids=None, progress=None) -> pd.DataFrame:
    """Run the catalogue (or the listed IDs). progress(i, n, test_id) is called before each test."""
    chosen = [t for t in TESTS if ids is None or t[0] in ids]
    rows = []
    for i, t in enumerate(chosen):
        if progress:
            progress(i, len(chosen), t[0])
        rows.append(run_test(t))
    return pd.DataFrame(rows)
