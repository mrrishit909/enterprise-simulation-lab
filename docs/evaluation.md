# Evaluation

Five businesses the demo never uses (seeds 21, 31, 41, 51, 61). Each is calibrated from its own 36 months of
history, then the demo's three arms go through 2,000 seeded scenarios of the calibrated agents and 2,000 runs of the true world
(customers in three hidden price-sensitivity tiers, the true parameters, the true competitor).

Reproduce with `docker compose run --rm test python -m es.evaluate`. The numbers below come from the Linux container, like the demo.
On macOS the same seeds give different histories (checked; most likely floating-point differences moving individual binomial draws), so
different numbers of the same kind; a replay is bit for bit on the platform that ran it.

## What 36 months of history recover

| Business | Notice-shock curvature psi (true 1.7) | Shock at a 5% rise (true 0.09) | Churn response to a 4% rise: model / truth | to a 12% rise: model / truth | Competitor reactions seen | Largest rise in history | Spreadsheet elasticity |
|---|---|---|---|---|---|---|---|
| 21 | 1.9 (90%: 1.1–2.7) | 0.121 | +14% / +15% | +111% / +96% | 3 of 7 | 10% | -0.79 |
| 31 | 2.3 (90%: 1.9–2.8) | 0.115 | +14% / +15% | +156% / +92% | 3 of 6 | 10% | +0.63 |
| 41 | 2.4 (90%: 1.9–3.0) | 0.090 | +14% / +16% | +136% / +104% | 1 of 5 | 10% | +0.01 |
| 51 | 1.3 (90%: 0.8–2.0) | 0.121 | +21% / +18% | +92% / +110% | 2 of 7 | 10% | -0.63 |
| 61 | 1.4 (90%: 0.8–2.1) | 0.115 | +18% / +17% | +83% / +105% | 3 of 10 | 10% | -0.47 |

## The decision: 12-month revenue change against the status quo, $M

| Business | Arm | Model mean (80% interval) | P(up), model | Truth mean | Model vs truth | Spreadsheet, pooled elasticity | Spreadsheet, no response | Month-12 monthly revenue change: model / truth |
|---|---|---|---|---|---|---|---|---|
| 21 | +12% | +2.74 (+1.74 to +3.68) | 100% | +2.50 | +10% | +8.32 | +7.75 | +72k / +27k |
| 21 | +12%, competitor -20% | +0.96 (+0.01 to +1.85) | 90% | +0.88 | +8% | +8.32 | +7.75 | -83k / -106k |
| 31 | +12% | +1.75 (+1.00 to +2.48) | 100% | +2.79 | -37% | +7.21 | +7.71 | -21k / +61k |
| 31 | +12%, competitor -20% | +0.19 (-0.45 to +0.82) | 64% | +1.25 | -85% | +7.21 | +7.71 | -153k / -67k |
| 41 | +12% | +2.48 (+1.76 to +3.14) | 100% | +2.75 | -10% | +7.44 | +7.45 | +42k / +56k |
| 41 | +12%, competitor -20% | +0.78 (+0.25 to +1.29) | 97% | +1.31 | -40% | +7.44 | +7.45 | -102k / -63k |
| 51 | +12% | +2.92 (+2.25 to +3.55) | 100% | +2.41 | +21% | +8.08 | +7.62 | +54k / +14k |
| 51 | +12%, competitor -20% | +1.30 (+0.76 to +1.82) | 100% | +0.86 | +52% | +8.08 | +7.62 | -81k / -112k |
| 61 | +12% | +3.59 (+2.91 to +4.21) | 100% | +2.94 | +22% | +8.15 | +7.82 | +123k / +67k |
| 61 | +12%, competitor -20% | +2.05 (+1.52 to +2.55) | 100% | +1.49 | +37% | +8.15 | +7.82 | -11k / -53k |

## Where the model's error sits

Each forecast of a year's revenue lands within about 3% of the truth. The change from a rise is a difference of two such
forecasts and only about 4% of revenue, so errors of a percent or two in each become tens of percent in the difference:

| Business | Status quo revenue: model vs truth | Status quo churned customers: model vs truth | +12% revenue: model vs truth |
|---|---|---|---|
| 21 | +0% | -1% | +0% |
| 31 | -0% | +1% | -2% |
| 41 | -1% | +4% | -1% |
| 51 | -1% | +4% | +0% |
| 61 | -0% | -1% | +1% |

## Is the spread honest?

Share of 2,000 true-world outcomes of the revenue change that fall inside the model's interval, over the ten business-arm pairs:

- inside the 80% interval: 58% (from 2% to 100%)
- inside the 50% interval: 32% (from 0% to 96%)

Run time 60 s on one core.
