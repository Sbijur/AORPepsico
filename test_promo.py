"""pytest test_promo.py  -  automated tests for the Mini PromoAI engine.

Part 1  Original unit tests (U1-U9): MILP vs brute force, infeasibility, budget, gap.
Part 2  Round-2 tests (F, M, B, E): one pytest case per catalogue entry in promo_tests.py.
Part 3  Scenario tests (SC1-SC7, WI1-WI6) exactly as shown in the app's "Test cases" tab.
Part 4  App smoke test (A01): app.py runs headless with no exceptions.

Run everything:            pytest test_promo.py -v
Only the fast formula set: pytest test_promo.py -v -k "F0"
"""
import os
from dataclasses import replace

import pytest

from promo_engine import *
from promo_tests import TESTS, run_test

products = DEFAULT_PRODUCTS
cfg = Config()


# ============================================================ Part 1: original unit tests (U1-U9)
def test_self_checks_pass_for_every_window():                      # U1
    for w in (1, 3, 5, 7):
        for name, passed, detail in run_self_checks(products, replace(cfg, window=w)):
            assert passed, f"window={w}: {name}: {detail}"

def test_optimizer_is_best_among_manual_calendars():                # U2
    df, _, _ = compare_scenarios(products, Config())
    f = df[df["Feasible"]]
    assert f.loc[f["Objective"].idxmax(), "Scenario"].startswith("G. OPTIMIZER")

def test_infeasible_calendar_is_flagged():                          # U3
    r = evaluate(products, Config(max_concurrent=2), [(0, 0.2)] * 3)
    assert not r.kpis["feasible"] and "Slot capacity" in r.kpis["violations"][0]

def test_impossible_constraints_return_none():                      # U4
    assert optimize(products, Config(min_margin_ratio=0.9)).best is None

def test_trade_budget_is_respected():                               # U5
    assert optimize(products, Config(trade_budget=300.0)).best.kpis["trade_spend"] <= 300.0 + 1e-9

def test_window_sweep_gain_is_positive():                           # U6
    assert (window_sweep(products, Config())["Gain vs no promo ($)"] > 0).all()

def test_milp_matches_bruteforce():                                 # U7
    for c in (Config(), Config(window=7), Config(max_concurrent=1), Config(min_margin_ratio=0.32), Config(trade_budget=500.0)):
        m, bf = optimize(products, c), optimize_bruteforce(products, c)
        assert m.info["proven_optimal"]
        assert abs(m.best.kpis["objective"] - bf.best.kpis["objective"]) < 1e-6, c

def test_milp_detects_infeasible_just_above_baseline_margin():     # U8 (was a duplicate of U4)
    # No promotion gives a 35.25% margin and every promotion lowers it, so a 36% floor has no feasible
    # calendar. This is the boundary case; U4 uses an extreme 90% floor.
    assert evaluate(products, cfg, [NO_PROMO] * 3).kpis["margin_ratio"] < 0.36
    assert optimize(products, Config(min_margin_ratio=0.36)).best is None

def test_milp_gap_tolerance_is_respected():                         # U9
    m, bf = optimize(products, Config(mip_gap=0.05)), optimize_bruteforce(products, Config())
    assert m.best.kpis["objective"] >= bf.best.kpis["objective"] * 0.95


# ============================================================ Part 2: round-2 tests (F, M, B, E)
@pytest.mark.parametrize("test", TESTS, ids=[f"{t[0]}-{t[2][:40]}" for t in TESTS])
def test_round2(test):
    r = run_test(test)
    assert r["Result"] == "PASS", f"{r['ID']} {r['Test']}: expected {r['Expected']}; got {r['Observed']}"


# ============================================================ Part 3: scenario tests shown in the app
EXPECTED_SC = {   # scenario -> (feasible?, must be the best feasible?)
    "A. No promotion (baseline)": (True, False),
    "B. Everyone promotes together (day 1)": (False, False),
    "C. One product per week, mid discount": (True, False),
    "D. One product per week, deep discount": (True, False),
    "E. One product per week, light discount": (True, False),
    "F. Back-to-back at start, mid discount": (True, False),
    "G. OPTIMIZER (MILP)": (True, True),
}

def test_scenarios_sc1_to_sc7():
    df, _, _ = compare_scenarios(products, cfg)
    for _, r in df.iterrows():
        feas, best = EXPECTED_SC[r["Scenario"]]
        assert bool(r["Feasible"]) == feas, r["Scenario"]
        assert bool(r["Best feasible?"]) == best, r["Scenario"]

def test_whatifs_wi1_to_wi6_move_the_right_way():
    w = constraint_whatifs(products, cfg).set_index("Rule set")
    obj = lambda k: 0.6 * w.loc[k, "PepsiCo profit"] + 0.4 * w.loc[k, "Retailer profit"]
    base = obj("Your current settings")
    assert obj("No slot limit (3 at once)") >= base - 1e-6           # WI1 relax
    assert obj("Strict slots (1 at a time)") <= base + 1e-6          # WI2 tighten
    assert obj("No margin floor") >= base - 1e-6                     # WI3 relax
    assert obj("Tight margin floor (32%)") <= base + 1e-6            # WI4 tighten
    assert w.loc["PepsiCo-only objective", "PepsiCo profit"] >= w.loc["Your current settings", "PepsiCo profit"] - 1e-6   # WI5
    assert w.loc["Retailer-friendly objective (30/70)", "Retailer profit"] >= w.loc["Your current settings", "Retailer profit"] - 1e-6  # WI6


# ============================================================ Part 4: app smoke test (A01)
def test_streamlit_app_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    at = AppTest.from_file(app, default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
