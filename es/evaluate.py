"""Held-out evaluation: five businesses the demo never uses. Each is calibrated the way the demo calibrates (36 months of
history, nothing else), run through the demo's three arms, and compared with the true world and with two spreadsheets.

    docker compose run --rm -T test python -m es.evaluate > docs/evaluation.md
"""
import time

import numpy as np

from . import engine as E, world as W

SEEDS = (21, 31, 41, 51, 61)
RUNS = 2000
ARMS = [{"name": "Status quo", "price_change": 0.0, "month": 1, "competitor": "reactive"},
        {"name": "+12%", "price_change": 0.12, "month": 1, "competitor": "reactive"},
        {"name": "+12%, competitor -20%", "price_change": 0.12, "month": 1, "competitor": "discount", "discount": 0.2, "discount_start": 2, "discount_months": 3}]


def m(v):
    return f"{v / 1e6:+.2f}"


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}%"


def main():
    t0 = time.time()
    out = ["# Evaluation", "", f"Five businesses the demo never uses (seeds {', '.join(map(str, SEEDS))}). Each is calibrated from its own 36 months of",
           f"history, then the demo's three arms go through {RUNS:,} seeded scenarios of the calibrated agents and {RUNS:,} runs of the true world",
           "(customers in three hidden price-sensitivity tiers, the true parameters, the true competitor).", "",
           "Reproduce with `docker compose run --rm test python -m es.evaluate`. The numbers below come from the Linux container, like the demo.",
           "On macOS the same seeds give different histories (checked; most likely floating-point differences moving individual binomial draws), so",
           "different numbers of the same kind; a replay is bit for bit on the platform that ran it.", ""]
    fits, rows, cover, base = [], [], [], []
    for seed in SEEDS:
        b, cal = W.business(seed), E.calibrate_seed(seed)
        st, pt = E.current_state(b), E.point(cal)
        w, grid = np.asarray(cal["churn"]["weights"]), np.asarray(cal["churn"]["psi_grid"])
        lo, hi = grid[np.searchsorted(np.cumsum(w), [0.05, 0.95])]
        fit_r = E.rise_response(pt, st["n"][..., None], st["P"], st["C"], [0.04, 0.12])
        true_r = E.rise_response(W.true_params(b), b["n_true"], b["P_now"], b["C_now"], [0.04, 0.12])
        co = cal["competitor"]
        fits.append((seed, pt["psi"], lo, hi, pt["kappa"], fit_r, true_r, co, cal["spreadsheet_elasticity"], cal["max_rise"]))
        ex = E.experiment(cal, st, ARMS, 12, RUNS, seed)
        tr = E.truth(seed, ARMS, 12, RUNS)
        sheet, flat = E.spreadsheet(b, ARMS, 12, cal["spreadsheet_elasticity"]), E.spreadsheet(b, ARMS, 12, 0.0)
        base.append((seed, ex["revenue"][:, 0].mean(), tr["revenue"][:, 0].mean(), ex["churned"][:, 0].mean(), tr["churned"][:, 0].mean()))
        for a in (1, 2):
            d, td = ex["revenue"][:, a] - ex["revenue"][:, 0], tr["revenue"][:, a] - tr["revenue"][:, 0]
            dm, tdm = ex["monthly"][:, a, -1] - ex["monthly"][:, 0, -1], tr["monthly"][:, a, -1] - tr["monthly"][:, 0, -1]
            rows.append((seed, ARMS[a]["name"], d, td, dm, tdm, sheet[a]["revenue"] - sheet[0]["revenue"], flat[a]["revenue"] - flat[0]["revenue"],
                         ex["revenue"][:, a].mean(), tr["revenue"][:, a].mean()))
            q = np.quantile(d, [0.1, 0.9, 0.25, 0.75])
            cover.append((np.mean((td >= q[0]) & (td <= q[1])), np.mean((td >= q[2]) & (td <= q[3]))))
    out += ["## What 36 months of history recover", "",
            "| Business | Notice-shock curvature psi (true 1.7) | Shock at a 5% rise (true 0.09) | Churn response to a 4% rise: model / truth | to a 12% rise: model / truth | Competitor reactions seen | Largest rise in history | Spreadsheet elasticity |",
            "|---|---|---|---|---|---|---|---|"]
    for seed, psi, lo, hi, kap, fr, tr_, co, el, mx in fits:
        out.append(f"| {seed} | {psi:.1f} (90%: {lo:.1f}–{hi:.1f}) | {kap:.3f} | {fr[0]:+.0%} / {tr_[0]:+.0%} | {fr[1]:+.0%} / {tr_[1]:+.0%} | {co['reactions']} of {len(co['rises'])} | {mx:.0%} | {el:+.2f} |")
    out += ["", "## The decision: 12-month revenue change against the status quo, $M", "",
            "| Business | Arm | Model mean (80% interval) | P(up), model | Truth mean | Model vs truth | Spreadsheet, pooled elasticity | Spreadsheet, no response | Month-12 monthly revenue change: model / truth |",
            "|---|---|---|---|---|---|---|---|---|"]
    for seed, name, d, td, dm, tdm, sh, fl, _, _ in rows:
        q = np.quantile(d, [0.1, 0.9])
        out.append(f"| {seed} | {name} | {m(d.mean())} ({m(q[0])} to {m(q[1])}) | {np.mean(d > 0):.0%} | {m(td.mean())} | {pct(d.mean(), td.mean())} | {m(sh)} | {m(fl)} | {dm.mean() / 1e3:+,.0f}k / {tdm.mean() / 1e3:+,.0f}k |")
    out += ["", "## Where the model's error sits", "",
            "Each forecast of a year's revenue lands within about 3% of the truth. The change from a rise is a difference of two such",
            "forecasts and only about 4% of revenue, so errors of a percent or two in each become tens of percent in the difference:", "",
            "| Business | Status quo revenue: model vs truth | Status quo churned customers: model vs truth | +12% revenue: model vs truth |", "|---|---|---|---|"]
    for (seed, bm, bt, cm, ct), r in zip(base, rows[0::2]):
        out.append(f"| {seed} | {pct(bm, bt)} | {pct(cm, ct)} | {pct(r[8], r[9])} |")
    c80, c50 = np.mean([c[0] for c in cover]), np.mean([c[1] for c in cover])
    out += ["", "## Is the spread honest?", "",
            f"Share of {RUNS:,} true-world outcomes of the revenue change that fall inside the model's interval, over the ten business-arm pairs:", "",
            f"- inside the 80% interval: {c80:.0%} (from {min(c[0] for c in cover):.0%} to {max(c[0] for c in cover):.0%})",
            f"- inside the 50% interval: {c50:.0%} (from {min(c[1] for c in cover):.0%} to {max(c[1] for c in cover):.0%})",
            "", f"Run time {time.time() - t0:.0f} s on one core."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
