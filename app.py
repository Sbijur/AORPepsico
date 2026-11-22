"""
Mini PromoAI  -  a 3-week, 3-product promotion planning simulator (Streamlit)
Run:  streamlit run app.py
"""
from dataclasses import replace

import inspect

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from promo_engine import (DEFAULT_PRODUCTS, NO_PROMO, Config, Product, compare_scenarios,
                          constraint_whatifs, describe, evaluate, optimize_bruteforce,
                          run_self_checks, window_sweep)
from promo_tests import GROUPS, TESTS, run_test

# works on old and new Streamlit versions (use_container_width is deprecated in newer ones)
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters
           else {"use_container_width": True})

st.set_page_config(page_title="Mini PromoAI", page_icon="🥤", layout="wide")
st.title("🥤 Mini PromoAI: 3-week promotion planner")
st.caption("A teaching-size version of the PromoAI system in the PepsiCo paper "
           "(Llenas et al.): demand model → optimizer → business constraints.")

# ------------------------------------------------------------------ sidebar: all the knobs
sb = st.sidebar
sb.header("1. Promotion window")
window = sb.slider("Promotion length x (days)", 1, 7, 3,
                   help="Every promotion runs for exactly x consecutive days.")
depth_opts = sb.multiselect("Discount depths the optimizer may choose", [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
                            default=[0.10, 0.20, 0.30], format_func=lambda v: f"{v:.0%}")

sb.header("2. Business rules (constraints)")
max_conc = sb.slider("Max products on promo on the same day", 1, 3, 2,
                     help="Retailer 'slot' capacity (paper: calendar/execution constraints).")
min_margin = sb.slider("Min PepsiCo margin % (profit ÷ revenue)", 0, 40, 30,
                       help="Margin-preservation constraint, eq. (10a) in the paper.") / 100
use_budget = sb.checkbox("Cap total trade spend")
budget = sb.number_input("Trade-spend budget ($)", 0.0, 5000.0, 600.0, 50.0) if use_budget else None

sb.header("3. Objective")
w_pep = sb.slider("Weight on PepsiCo profit (rest = retailer profit)", 0.0, 1.0, 0.6, 0.1,
                  help="Paper eq. (2): ω1 / ω2 weights between PepsiCo and retailer.")
funding = sb.slider("Share of discount funded by PepsiCo", 0.0, 1.0, 0.4, 0.05)

sb.header("4. Demand behaviour")
cann = sb.slider("Cannibalization strength", 0.0, 1.0, 0.5, 0.05,
                 help="Sales lost per unit of average discount on the OTHER products same day.")
dip = sb.slider("Post-promo dip (pantry loading)", 0.0, 0.5, 0.20, 0.05,
                help="Demand drop for x days after a promo ends.")
ev_start = sb.slider("Demand event starts on day", 1, 21, 12)
ev_len = sb.slider("Event length (days)", 1, 5, 3)
ev_boost = sb.slider("Event demand boost", 0.0, 1.0, 0.30, 0.05)

sb.header("5. MILP solver")
gap = sb.selectbox("Optimality gap tolerance", [0.0, 0.01, 0.05], format_func=lambda g: "0% (prove optimal)" if g == 0 else f"{g:.0%}",
                   help="The paper stops its MILP at a 1-5% gap in production for faster turnaround.")

if not depth_opts:
    st.error("Pick at least one discount depth in the sidebar.")
    st.stop()
if len(depth_opts) > 4:
    st.warning("More than 4 depths makes the solver slow - keeping the first 4.")
    depth_opts = depth_opts[:4]

with st.expander("🧺 Product assumptions (editable)", expanded=False):
    prod_df = pd.DataFrame([vars(p) for p in DEFAULT_PRODUCTS]).rename(columns={
        "name": "Product", "base_units": "Base units/day", "shelf_price": "Shelf price",
        "wholesale": "Wholesale (sell-in)", "unit_cost": "Unit cost", "elasticity": "Promo elasticity"})
    edited = st.data_editor(prod_df, hide_index=True, **STRETCH, disabled=["Product"])
products = tuple(Product(r["Product"], float(r["Base units/day"]), float(r["Shelf price"]),
                         float(r["Wholesale (sell-in)"]), float(r["Unit cost"]),
                         float(r["Promo elasticity"])) for _, r in edited.iterrows())

cfg = Config(window=window, depths=tuple(sorted(depth_opts)), cannibalization=cann, dip=dip,
             event_start=ev_start, event_len=ev_len, event_boost=ev_boost, funding_share=funding,
             w_pepsico=w_pep, max_concurrent=max_conc, min_margin_ratio=min_margin, trade_budget=budget,
             mip_gap=gap)


# ------------------------------------------------------------------ cached heavy calls
@st.cache_data(show_spinner="Solving the MILP…")
def _compare(products, cfg):
    return compare_scenarios(products, cfg)


@st.cache_data(show_spinner="Brute-force cross-check of every calendar…")
def _landscape(products, cfg):
    return optimize_bruteforce(products, cfg)


@st.cache_data(show_spinner="Re-optimising under different rules…")
def _whatifs(products, cfg):
    return constraint_whatifs(products, cfg)


@st.cache_data(show_spinner="Sweeping window length 1-7 days…")
def _sweep(products, cfg):
    return window_sweep(products, cfg)


@st.cache_data(show_spinner="Running tests…")
def _checks(products, cfg):
    return run_self_checks(products, cfg)


scen_df, opt, results = _compare(products, cfg)
if opt.best is None:
    st.error("No calendar satisfies all constraints. Relax the slot limit, margin floor or budget.")
    st.stop()
best = opt.best
land = _landscape(products, cfg)       # brute force: ONLY for validation + histogram
base = evaluate(products, cfg, [NO_PROMO] * 3)

tabs = st.tabs(["① How it works", "② Optimal calendar", "③ Test cases", "④ Window sweep", "⑤ Automated tests"])

# ================================================================== TAB 1
with tabs[0]:
    st.subheader("The simulation in one page")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Setting:** 3 products × **21 days** (3 weeks). Each product may run **one promotion of exactly
{cfg.window} day(s)**, at a depth chosen by the optimizer. We also simulate {cfg.window} extra "tail" days so
the post-promo dip can't be dodged by promoting at the very end.

**Decision:** for each product → *no promo*, or *(start day, discount depth)*.
That is **{opt.info["n_binary"]} yes/no (binary) variables** in the MILP; the search space has
{opt.n_total:,} possible calendars, but the solver never lists them - it uses LP relaxations and
branch-and-bound ({opt.info["nodes"]} nodes this run, {opt.info["solve_time"]:.1f}s).

**Demand model** (replaces the paper's LightGBM forecast):
""")
        st.latex(r"units = base \times season \times (1-d)^{\varepsilon} \times (1-dip\cdot postPromo) \times (1-\kappa\cdot avgOtherDiscount)")
        st.markdown("""
| Term | Meaning | Paper analogue |
|---|---|---|
| (1-d)^ε | price response: deeper discount → more units | LightGBM lift curve / PWL approximation |
| season | demand event (e.g. game weekend) | holiday / Super Bowl features |
| dip | pantry loading after a promo | min-spacing rule rationale |
| κ · avg other discount | cannibalization | "competitive discount pressure", eq. (5) |
""")
    with c2:
        st.markdown("""
**Money model**

- PepsiCo funds a share *f* of the shelf discount (trade spend): `unit revenue = wholesale − f·price·d`
- PepsiCo profit = units × (unit revenue − unit cost)
- Retailer profit = units × (shelf price·(1−d) − PepsiCo unit revenue)

**Objective** (paper eq. 2, simplified):
""")
        st.latex(r"\max\;\; \omega\cdot Profit_{PepsiCo} + (1-\omega)\cdot Profit_{Retailer}")
        st.markdown("""
**Constraints** (checked for every calendar):
1. **Slot capacity** – at most *N* products on promo the same day
2. **Margin floor** – PepsiCo profit ÷ revenue ≥ γ (eq. 10a)
3. **Trade budget** *(optional)* – total trade spend ≤ B
4. **Window length** – built into the options: every promo lasts exactly *x* days

**Optimizer: a real MILP** (HiGHS via `scipy.optimize.milp`; the paper uses Gurobi). Brute force is kept
only to double-check the MILP on this tiny problem - it cannot scale to real PromoAI sizes.
""")
    with st.expander("📐 The MILP, step by step", expanded=False):
        st.markdown("""
**Variables**  `x[p,o]` ∈ {0,1}: product *p* uses option *o* (no promo, or start day + depth).
`y[a,o,b,o′]` ∈ [0,1]: equals `x[a,o]·x[b,o′]` ("*a* does *o* while *b* does *o′*").

**Why the extra `y`?** Cannibalization means product *a*'s sales depend on product *b*'s discount, so profit
contains a **product of two binary choices**. A MILP must be linear, so each product is replaced by `y`, with
linking constraints `Σ_o′ y[a,o,b,o′] = x[a,o]` and `Σ_o y[a,o,b,o′] = x[b,o′]` that force `y = x·x′` exactly.

**Model**
""")
        st.latex(r"\max\; \omega\,\Pi^{P}(x,y) + (1-\omega)\,\Pi^{R}(x,y)")
        st.markdown("""
| Constraint | Written as |
|---|---|
| One option per product | Σ_o x[p,o] = 1 |
| Slot capacity (each day t) | Σ_p Σ_{o on promo on day t} x[p,o] ≤ N |
| Margin floor | Π^P − γ·Revenue^P ≥ 0 |
| Trade budget (optional) | TradeSpend ≤ B |
| Linking | y = x·x′ (above) |

The promotion length *x* days and the price response (1−d)^ε are baked into each option's daily numbers,
so no piecewise-linear step is needed (the paper needs PWL because it embeds a non-linear LightGBM curve).
""")
    st.info("Tip: change a slider and watch tabs ②–④ update. Start with the promotion length x and the slot limit.")

# ================================================================== TAB 2
with tabs[1]:
    st.subheader("Optimal promotion calendar")
    st.success(describe(products, best.schedule))
    k, b = best.kpis, base.kpis
    inf = opt.info
    bf_obj = land.best.kpis["objective"]
    rel = (bf_obj - k["objective"]) / bf_obj * 100
    st.caption(f"🔧 **MILP solver:** {inf['status'].split('(')[-1].rstrip(')')} · {inf['solve_time']:.1f}s · "
               f"{inf['n_binary']} binary + {inf['n_continuous']:,} continuous variables · {inf['n_constraints']} constraints · "
               f"{inf['nodes']} branch-and-bound nodes · gap {0 if inf['gap'] is None else inf['gap']:.2%}")
    if gap == 0:
        (st.success if abs(rel) < 1e-6 else st.error)(
            f"✅ Cross-check: brute force over all {land.n_total:,} calendars finds the same optimum "
            f"({bf_obj:,.2f})." if abs(rel) < 1e-6 else f"MILP differs from brute force by {rel:.4f}%")
    else:
        st.info(f"With a {gap:.0%} gap tolerance the MILP may stop early. Here it is {rel:.3f}% below the true optimum "
                f"({bf_obj:,.2f}, from brute force).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PepsiCo profit", f"${k['pepsico_profit']:,.0f}", f"{k['pepsico_profit'] - b['pepsico_profit']:+,.0f} vs no promo")
    m2.metric("Retailer profit", f"${k['retailer_profit']:,.0f}", f"{k['retailer_profit'] - b['retailer_profit']:+,.0f} vs no promo")
    m3.metric("Units sold", f"{k['units']:,.0f}", f"{k['units'] - b['units']:+,.0f} vs no promo")
    m4.metric("Trade spend", f"${k['trade_spend']:,.0f}", f"margin {k['margin_ratio']:.1%}", delta_color="off")

    # Gantt
    rows = [dict(Product=p.name, Start=s + 1, End=s + 1 + cfg.window, Depth=f"{d:.0%}")
            for p, (s, d) in zip(products, best.schedule) if s is not None]
    ev = pd.DataFrame([dict(Start=cfg.event_start, End=cfg.event_start + cfg.event_len)])
    xs = alt.Scale(domain=[1, cfg.days + 1], nice=False)
    event_layer = alt.Chart(ev).mark_rect(opacity=0.15, color="orange").encode(
        x=alt.X("Start:Q", scale=xs, title="Day"), x2="End:Q")
    if rows:
        bars = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadius=4, height=28).encode(
            x=alt.X("Start:Q", scale=xs, title="Day"), x2="End:Q",
            y=alt.Y("Product:N", sort=[p.name for p in products], title=None),
            color=alt.Color("Depth:N", title="Discount"), tooltip=["Product", "Start", "End", "Depth"])
        chart = event_layer + bars
    else:
        chart = event_layer
    st.altair_chart(chart.properties(height=170, title="Promotion calendar (orange band = demand event)"),
                    **STRETCH)

    # daily units
    d1 = best.daily.assign(Scenario="Optimized")
    d0 = base.daily.assign(Scenario="No promotion")
    dd = pd.concat([d0, d1])
    line = alt.Chart(dd).mark_line().encode(
        x=alt.X("day:Q", title="Day (days > 21 are the post-promo tail)"), y=alt.Y("units:Q", title="Units/day"),
        color=alt.Color("Scenario:N", scale=alt.Scale(range=["#aaaaaa", "#d62728"])),
        strokeDash=alt.StrokeDash("Scenario:N")).properties(height=190, width=260)
    st.altair_chart(line.facet(column=alt.Column("product:N", title=None)), **STRETCH)
    st.caption("Look for: spike during the promo, dip right after (pantry loading), and lower units on a product "
               "when another one is on promo the same day (cannibalization).")

    # where does the optimum sit in the search space
    hist_df = pd.DataFrame({"objective": land.feasible_objectives})
    h = alt.Chart(hist_df).mark_bar(opacity=0.7).encode(
        x=alt.X("objective:Q", bin=alt.Bin(maxbins=60), title="Objective value of a calendar"),
        y=alt.Y("count()", title="# of feasible calendars"))
    marks = pd.DataFrame({"v": [b["objective"], k["objective"]], "label": ["No promo", "Optimal"]})
    rules = alt.Chart(marks).mark_rule(strokeWidth=3).encode(
        x="v:Q", color=alt.Color("label:N", scale=alt.Scale(range=["#555555", "#d62728"])))
    st.altair_chart((h + rules).properties(height=220, title="All feasible calendars: where does the optimum sit?"),
                    **STRETCH)
    fo = land.feasible_objectives
    pct_worse = (fo < b["objective"]).mean() * 100
    share = (np.median(fo) - b["objective"]) / (k["objective"] - b["objective"]) * 100
    st.caption(f"{pct_worse:.0f}% of the {land.n_feasible:,} feasible calendars are worse than running no promotions. "
               f"The median feasible calendar captures about {share:.0f}% of the optimal gain, so timing and depth "
               "are where the optimizer earns its keep. (Lower the elasticities in the product table to see "
               "promotions start to lose money.)")

    with st.expander("Day-by-day table of the optimal calendar"):
        piv = best.daily.pivot(index="day", columns="product", values="units").round(1)
        disc = best.daily.pivot(index="day", columns="product", values="discount").add_suffix(" disc")
        st.dataframe(piv.join(disc.apply(lambda c: c.map(lambda v: f"{v:.0%}" if v else ""))), **STRETCH)

# ================================================================== TAB 3
with tabs[2]:
    st.subheader("Test cases: manual calendars vs the MILP optimizer")
    st.markdown("Every scenario is simulated with **your current settings**. A calendar that breaks a rule is "
                "**infeasible** and can't win, no matter how high its profit.")
    show = scen_df.copy()
    show["Result"] = np.where(show["Best feasible?"], "🏆 OPTIMAL",
                              np.where(show["Feasible"], "feasible", "❌ infeasible"))
    # Test IDs SC1-SC7 (wiki 3.3 A) with the property each calendar must satisfy under ANY settings:
    # the MILP must be the best feasible calendar, and no manual calendar may beat it.
    sc_id = {n: f"SC{i + 1}" for i, n in enumerate(show["Scenario"])}
    milp_obj = show.loc[show["Scenario"].str.startswith("G."), "Objective"]
    milp_obj = float(milp_obj.iloc[0]) if len(milp_obj) else np.nan

    def _sc_expect(r):
        if r["Scenario"].startswith("G."):
            return "Best feasible calendar"
        if r["Scenario"].startswith("A."):
            return "Feasible baseline, ≤ MILP" if r["Feasible"] else "Baseline breaks a rule"
        return "≤ MILP if feasible; flagged if it breaks a rule"

    def _sc_pass(r):
        if r["Scenario"].startswith("G."):
            return bool(r["Best feasible?"])
        if not r["Feasible"]:
            return bool(r["Why infeasible"])                     # infeasible must name the rule
        return r["Objective"] <= milp_obj + 1e-6

    show.insert(0, "ID", show["Scenario"].map(sc_id))
    show["Expected"] = show.apply(_sc_expect, axis=1)
    show["Test"] = np.where(show.apply(_sc_pass, axis=1), "✅ Pass", "❌ Fail")
    cols = ["ID", "Scenario", "Result", "Expected", "Test", "Objective", "PepsiCo profit", "Retailer profit",
            "Trade spend", "Units", "Margin %", "Max promos/day", "Why infeasible"]
    st.dataframe(show[cols].style.format({"Objective": "{:,.1f}", "PepsiCo profit": "{:,.0f}",
                                          "Retailer profit": "{:,.0f}", "Trade spend": "{:,.0f}",
                                          "Units": "{:,.0f}", "Margin %": "{:.1f}"})
                 .apply(lambda r: ["background-color:#d4edda;color:#14532d" if r["Result"] == "🏆 OPTIMAL"
                                   else "background-color:#f8d7da;color:#7f1d1d" if r["Result"] == "❌ infeasible" else ""
                                   for _ in r], axis=1),
                 hide_index=True, **STRETCH)
    ch = alt.Chart(show).mark_bar().encode(
        y=alt.Y("Scenario:N", sort=None, title=None), x=alt.X("Objective:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Result:N", scale=alt.Scale(domain=["🏆 OPTIMAL", "feasible", "❌ infeasible"],
                                                    range=["#2ca02c", "#9ecae1", "#e6550d"])),
        tooltip=["Scenario", "Objective", "Result"]).properties(height=260)
    st.altair_chart(ch, **STRETCH)
    winner = show.loc[show["Best feasible?"], "Scenario"].tolist()
    runner = show[show["Feasible"] & ~show["Best feasible?"] & ~show["Scenario"].str.startswith("A.")]
    if winner and len(runner):
        r = runner.loc[runner["Objective"].idxmax()]
        gap = show.loc[show["Best feasible?"], "Objective"].iloc[0] - r["Objective"]
        st.markdown(f"**Verdict:** *{winner[0]}* gives the best feasible result. Best manual rule-of-thumb: "
                    f"*{r['Scenario']}*, which leaves **${gap:,.0f}** of objective value on the table.")

    n_sc = int((show["Test"] == "✅ Pass").sum())
    st.caption(f"Scenario tests SC1–SC{len(show)}: **{n_sc} / {len(show)} pass** under your current settings.")

    st.markdown("#### What does each business rule cost? (re-optimised under different rules)")
    wi = _whatifs(products, cfg)
    # Test IDs WI1-WI6 (wiki 3.3 B): relaxing a rule must never lower the optimum, tightening must never raise it.
    # Whether a variant is a relaxation or a tightening depends on your current settings.
    wi_rules = {
        "No slot limit (3 at once)": ("WI1", "max_concurrent", 3),
        "Strict slots (1 at a time)": ("WI2", "max_concurrent", 1),
        "No margin floor": ("WI3", "min_margin_ratio", 0.0),
        "Tight margin floor (32%)": ("WI4", "min_margin_ratio", 0.32),
        "PepsiCo-only objective": ("WI5", "w_pepsico", 1.0),
        "Retailer-friendly objective (30/70)": ("WI6", "w_pepsico", 0.3),
    }
    cur = wi.set_index("Rule set").loc["Your current settings"] if "PepsiCo profit" in wi else None
    obj = lambda r: cfg.w_pepsico * r["PepsiCo profit"] + (1 - cfg.w_pepsico) * r["Retailer profit"]
    ids, exps, verdicts = [], [], []
    for _, r in wi.iterrows():
        name = r["Rule set"]
        if name not in wi_rules or cur is None:
            ids.append("–"); exps.append("Reference row"); verdicts.append(""); continue
        tid, field, val = wi_rules[name]
        ids.append(tid)
        if r.get("Schedule") == "INFEASIBLE" or pd.isna(r.get("PepsiCo profit")):
            tighter = (field == "max_concurrent" and val < cfg.max_concurrent) or \
                      (field == "min_margin_ratio" and val > cfg.min_margin_ratio)
            exps.append("May become infeasible (tighter rule)" if tighter else "Must stay feasible")
            verdicts.append("✅ Pass" if tighter else "❌ Fail"); continue
        if field == "w_pepsico":
            party = "PepsiCo profit" if val > cfg.w_pepsico else "Retailer profit"
            if val == cfg.w_pepsico:
                exps.append("Same as current"); ok = abs(obj(r) - obj(cur)) < 1e-6
            else:
                exps.append(f"{party} not lower than current"); ok = r[party] >= cur[party] - 1e-6
        else:
            cur_val = getattr(cfg, field)
            relax = (val > cur_val) if field == "max_concurrent" else (val < cur_val)
            if val == cur_val:
                exps.append("Same as current"); ok = abs(obj(r) - obj(cur)) < 1e-6
            elif relax:
                exps.append("Relaxed: objective not lower"); ok = obj(r) >= obj(cur) - 1e-6
            else:
                exps.append("Tightened: objective not higher"); ok = obj(r) <= obj(cur) + 1e-6
        verdicts.append("✅ Pass" if ok else "❌ Fail")
    wi.insert(0, "ID", ids)
    wi.insert(2, "Expected", exps)
    wi.insert(3, "Test", verdicts)
    st.dataframe(wi.style.format({c: "{:,.1f}" for c in wi.columns
                                  if c not in ("ID", "Rule set", "Expected", "Test", "Schedule")}, na_rep="–"),
                 hide_index=True, **STRETCH)
    n_wi = sum(v == "✅ Pass" for v in verdicts)
    st.caption(f"What-if tests WI1–WI6: **{n_wi} / 6 pass**. Lower profit than 'Your current settings' = that rule "
               "is *binding*. If a row equals it, the rule isn't constraining the answer.")

# ================================================================== TAB 4
with tabs[3]:
    st.subheader("How long should a promotion run?")
    sw = _sweep(products, cfg)
    ok = sw.dropna(subset=["Objective"])
    if ok.empty:
        st.error("No feasible calendars for any window.")
    else:
        c1, c2 = st.columns(2)
        gain = alt.Chart(ok).mark_line(point=True).encode(
            x=alt.X("Window (days):O"), y=alt.Y("Gain vs no promo ($):Q")).properties(
            height=260, title="Optimal gain vs no-promo, by promotion length")
        c1.altair_chart(gain, **STRETCH)
        units = alt.Chart(ok).mark_line(point=True, color="#2ca02c").encode(
            x=alt.X("Window (days):O"), y=alt.Y("Units:Q", scale=alt.Scale(zero=False))).properties(
            height=260, title="Units sold at the optimum")
        c2.altair_chart(units, **STRETCH)
        st.dataframe(sw.style.format({c: "{:,.1f}" for c in sw.columns if c not in ("Window (days)", "Schedule")}),
                     hide_index=True, **STRETCH)
        bw = int(ok.loc[ok["Gain vs no promo ($)"].idxmax(), "Window (days)"])
        st.info(f"Under these settings the highest gain comes from a **{bw}-day** promotion. "
                "If gain keeps rising with x, no rule is stopping promotions from being longer - "
                "try tightening the margin floor or the trade budget to see an interior optimum.")

# ================================================================== TAB 5
with tabs[4]:
    st.subheader("Automated tests")
    st.markdown("Two layers of tests. The **self-checks** run on *your current settings*. The **round-2 suite** "
                "runs on the wiki base case, so every expected value can be checked by hand. All of them also run "
                "with `pytest test_promo.py -v`.")

    # ---------- A. self-checks T1-T9 on the current settings
    st.markdown("#### A. Self-checks T1–T9 (your current settings)")
    checks = _checks(products, cfg)
    n_ok = sum(c[1] for c in checks)
    st.dataframe(pd.DataFrame([{"ID": n.split()[0], "Check": n.split("  ", 1)[-1].strip(), "Observed": d,
                                "Result": "✅ Pass" if p else "❌ Fail"} for n, p, d in checks])
                 .style.apply(lambda r: ["background-color:#d4edda;color:#14532d" if r["Result"] == "✅ Pass"
                                         else "background-color:#f8d7da;color:#7f1d1d" for _ in r], axis=1),
                 hide_index=True, **STRETCH)
    st.metric("Self-checks passed", f"{n_ok} / {len(checks)}")

    # ---------- B. round-2 suite (base case)
    st.markdown("#### B. Round-2 test suite (wiki base case)")
    st.markdown(
        "| Group | IDs | What it checks |\n|---|---|---|\n"
        "| **Formula** | F01–F09 | Hand-checkable numbers: lift, money per unit, dip, event, cannibalisation, "
        "profits + costs = consumer spend, option counts, tail days |\n"
        "| **Metamorphic** | M01–M11 | Change an input, the answer must move predictably: product order, volume and "
        "currency scaling, event timing, option sets, budget, cannibalisation, dip, weights, binding floor |\n"
        "| **Boundary** | B01–B03 | Elasticity −2 to −4: where promotions switch on, MILP = brute force at every point |\n"
        "| **Edge** | E01–E10 | Wrong product count, very long promotions, zero slots, tiny time limit, 1% gap, "
        "determinism, violation messages, loss-making and zero-volume products |")
    c1, c2 = st.columns([3, 1])
    groups = c1.multiselect("Groups to run", list(GROUPS.values()), default=list(GROUPS.values()))
    chosen = [t for t in TESTS if t[1] in groups]
    est = {"Formula": 4, "Metamorphic": 80, "Boundary": 12, "Edge": 15}
    c2.metric("Tests selected", len(chosen), f"≈{sum(est[g] for g in groups)} s", delta_color="off")
    if st.button("▶ Run round-2 tests", type="primary", disabled=not chosen):
        bar = st.progress(0.0, text="Starting…")
        rows = []
        for i, t in enumerate(chosen):
            bar.progress(i / len(chosen), text=f"Running {t[0]}: {t[2]}")
            rows.append(run_test(t))
        bar.progress(1.0, text="Done")
        st.session_state["round2"] = pd.DataFrame(rows)
    r2 = st.session_state.get("round2")
    if r2 is not None:
        n_pass = int((r2["Result"] == "PASS").sum())
        m = st.columns(len(GROUPS) + 1)
        m[0].metric("Round-2 passed", f"{n_pass} / {len(r2)}")
        for col, g in zip(m[1:], GROUPS.values()):
            sub = r2[r2["Group"] == g]
            col.metric(g, f"{int((sub['Result'] == 'PASS').sum())} / {len(sub)}" if len(sub) else "–")
        view = r2.assign(Result=r2["Result"].map({"PASS": "✅ Pass", "FAIL": "❌ Fail"}))
        st.dataframe(view[["ID", "Group", "Test", "Expected", "Observed", "Result", "Seconds"]]
                     .style.apply(lambda r: ["background-color:#d4edda" if r["Result"] == "✅ Pass"
                                             else "background-color:#f8d7da" for _ in r], axis=1),
                     hide_index=True, **STRETCH)
        b01 = r2[r2["ID"] == "B01"]
        if len(b01) and b01["Result"].iloc[0] == "PASS":
            st.info("**Elasticity switch point:** " + b01["Observed"].iloc[0].split(";")[1].strip().split(" (")[0]
                    + ". Weaker elasticities make every promotion lose money, so the optimizer runs none.")
        if n_pass < len(r2):
            st.error("Some tests failed. The Observed column shows what went wrong.")
    else:
        st.caption("Press the button to run the selected groups. The metamorphic group re-solves the MILP many times.")

    st.markdown("#### C. Run everything from a terminal")
    st.code("pytest test_promo.py -v              # 9 unit tests + 33 round-2 tests + scenario tests + app smoke test\n"
            "pytest test_promo.py -v -k \"F0\"     # only the fast formula tests", language="bash")
