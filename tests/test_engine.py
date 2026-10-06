"""The simulator's bookkeeping, seeded replay, the fits and the competitor reader, without a database."""
import numpy as np

from es import engine as E, world as W

ARMS = [{"price_change": 0.0, "month": 1, "competitor": "reactive"}, {"price_change": 0.12, "month": 1, "competitor": "reactive"},
        {"price_change": 0.12, "month": 1, "competitor": "discount", "discount": 0.2, "discount_start": 2, "discount_months": 3}]


def test_customers_are_conserved_and_a_full_desk_saves_exactly_its_capacity():
    b = W.business(18)
    P, C, _ = W.plan_paths(ARMS, b["P_now"], b["C_now"], W.TRUE_COMPETITOR, np.random.default_rng(1), 12)
    o = W.simulate(b["n_true"], P, C, W.true_params(b), np.random.default_rng(2))
    start = b["n_true"].sum()
    assert np.allclose(o["n"].sum(axis=(1, 2, 3, 4)), start - o["churned"].sum(1) + o["signups"].sum(1))
    assert o["saturated"].any() and (o["saves"][o["saturated"]] == W.DESK["capacity"]).all() and (o["saves"] <= o["intents"]).all()
    assert (o["churned"][1, 0] > 1.5 * o["churned"][0, 0]) and o["revenue"][1, 0] > o["revenue"][0, 0]     # the notice shock, and the higher price


def test_a_scenario_replays_bit_for_bit_from_its_seed_alone():
    cal, st = E.calibrate_seed(18), E.current_state(W.business(18))
    a = E.run_one(cal, st, ARMS, 12, 2026, 41)[0]
    b = E.run_one(cal, st, ARMS, 12, 2026, 41)[0]
    c = E.run_one(cal, st, ARMS, 12, 2026, 42)[0]
    assert E.digest(a) == E.digest(b) and E.digest(a) != E.digest(c)
    ex = E.experiment(cal, st, ARMS, 12, 50, 2026)        # scenario 41 inside a batch is the same as on its own
    assert ex["hashes"][41] == E.digest(a)


def test_market_effects_undo_the_confounding_a_spreadsheet_falls_into():
    b, cal = W.business(18), E.calibrate_seed(18)
    pt = E.point(cal)
    assert cal["spreadsheet_elasticity"] < 0 < min(pt["beta"])           # pooled: dearer markets churn less; within markets: rises raise churn
    assert np.corrcoef(pt["u"], b["u"] - b["u"][0])[0, 1] > 0.9
    w, grid = np.asarray(cal["churn"]["weights"]), np.asarray(cal["churn"]["psi_grid"])
    lo, hi = grid[np.searchsorted(np.cumsum(w), [0.05, 0.95])]
    assert lo <= W.TRUE["psi"] <= hi and hi - lo >= 0.8                    # the truth is inside, and the interval is wide


def test_competitor_promotions_are_read_off_a_noisy_scrape():
    P = np.ones((2, 13))
    P[0, 4:] = 1.06                                                         # our rise in market 0, month 4
    C = np.ones((2, 13))
    C[0, 5:8] = 0.8                                                         # a reaction a month later
    C[1, 9:11] = 0.85                                                       # an unprovoked promotion
    obs = {"P": P, "C_obs": C * np.exp(np.random.default_rng(0).normal(0, W.SCRAPE_SD, C.shape))}
    ep = E.competitor_episodes(obs)
    assert ep["promotions"] == 2 and ep["reactions"] == 1 and ep["rises"] == [[0, 4]] and ep["durations"] == [3, 2]
    assert abs(ep["depths"][0] - 0.2) < 0.04


def test_spreadsheet_and_truth_agree_on_the_easy_part():
    b = W.business(18)
    sheet = E.spreadsheet(b, ARMS[:2], 12, 0.0)
    tr = E.truth(18, ARMS[:2], 12, 200)
    assert abs(sheet[0]["revenue"] / tr["revenue"][:, 0].mean() - 1) < 0.03          # with no change both project the same year
    assert sheet[1]["revenue"] - sheet[0]["revenue"] > 2 * (tr["revenue"][:, 1] - tr["revenue"][:, 0]).mean()   # but ignoring the response overstates a rise
