# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: one scenario of 59,733 customers, 3,456 cohort-months of
history, 129 agents, one experiment of 10,000 stored scenarios in three arms. Reproduce with `make demo`, then `make load-test`.
Other builds were running Docker on the same machine at the time, so these are lower bounds.

## Reads: the scenario overview and the experiment's distributions, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 449 | 44.9 req/s | 25.8 ms | 30.2 ms | 32.2 ms | 0.00% |
| 8 | 885 | 88.0 req/s | 87.3 ms | 115.7 ms | 128.3 ms | 0.00% |
| 32 | 764 | 74.0 req/s | 339.1 ms | 392.6 ms | 416.4 ms | 0.00% |

Target from the blueprint: p95 under 500 ms for reads: met at 32 concurrent clients. The scenario overview rebuilds the history
arrays from 3,456 cohort rows on every call; the distributions call computes P(revenue up) over the 10,000 stored runs in SQL.
Neither is cached.

## Compute (one core)

| Work | Time |
|---|---|
| Calibrate: 25 churn fits (one per curvature), signups, competitor episodes, and 200 draws for the response band | 0.17 s (API call) |
| One seeded run, three arms, 12 months, events and metrics stored | 18 ms (API call) |
| 10,000 seeded scenarios x 3 arms in the worker | 23 s, 2.3 ms a scenario |
| Replay of one scenario, events stored, hash checked | 34 ms (API call); the simulation itself 5 ms |
| Sobol analysis: 9,216 runs of two arms, the surrogate and its 40,000-point indices | 16.5 s (API call) |
| Truth for an experiment: 2,000 runs of the true world (three hidden tiers) | inside the batch job, not timed separately |
| Evaluation: five businesses, 2,000 model and 2,000 true runs each | 60 s |

The cost of a scenario is NumPy dispatch, not arithmetic: 96 cohorts per arm is a few hundred numbers per month. A batch splits
cleanly by scenario index across workers, since each scenario depends only on (entropy, index); that is not built.
Not measured: experiments larger than 20,000 scenarios (the API's limit), or many concurrent batches.
