"""The invented subscription business, its customers, its competitor and its retention desk, and the simulator that moves
them month by month. The same `simulate` runs the hidden true world (customers in three price-sensitivity tiers) and the
calibrated model (one tier, parameters fitted from history), so the only differences are the ones a real lab would have:
fitted parameters, an unseen mixture, and a competitor policy guessed from a handful of episodes.

Customers are agents grouped into cohorts (market x plan x tenure band x tier) and drawn binomially, which keeps 10,000
seeded scenarios to seconds. Prices and competitor prices are indices on the list price (1.0 = list).
"""
import functools

import numpy as np
from scipy.special import expit

SEGMENTS = ["starter", "team", "business"]
LIST_PRICE = np.array([29.0, 99.0, 399.0])            # $ a month
SEG_SHARE = np.array([0.62, 0.29, 0.09])
MARKETS = ["Northeast", "Southeast", "Midwest", "Texas", "Mountain", "California", "Pacific NW", "Canada"]
BANDS = ["0-3 months", "4-12 months", "13-24 months", "25+ months"]
BAND_SHARE = np.array([0.12, 0.25, 0.23, 0.40])
AGE = np.array([1 / 3, 1 / 9, 1 / 12])                # monthly share of a tenure band that moves on to the next
HISTORY_MONTHS, WARMUP = 36, 24
REF = 0.05                                            # the notice shock is expressed at a 5% rise
SCRAPE_SD = 0.02                                      # competitor prices come from a web scrape

# The true world. Nothing outside this module and the evaluation reads it.
TRUE = {"alpha": [-2.85, -3.35, -3.95], "tenure": [0.0, -0.4, -0.65, -0.85], "beta": [1.4, 1.1, 0.7], "gamma": 1.6,
        "kappa": 0.09, "psi": 1.7, "eta": [2.2, 1.6, 1.0],
        "mult": [0.4, 1.0, 2.2], "share": [0.3, 0.5, 0.2], "market_sd": 0.3}
TRUE_COMPETITOR = {"react_p": 0.45, "min_rise": 0.05, "depth": (0.10, 0.25), "months": (2, 4), "promo_rate": 0.03, "drift_p": 0.05}
DESK = {"save_rate": 0.25, "capacity": 560}           # the retention desk: an operating assumption the company knows


def simulate(n0, P, C, p, rng, panel=None):
    """Run cohorts forward. n0 (M,S,B,T) customers now; P, C (A,M,H+1) our and the competitor's price index, column 0 = now.
    p: alpha (S), u (M), ten (B), beta (S), gamma, kappa, psi, mult (T), share (T), lam (M,S), eta (S), save_rate, capacity.
    Returns monthly totals per arm, each (A,H). `panel`, a list, receives what billing and the CRM log for arm 0 each month:
    customers at risk, cancellation requests and saves per cohort, signups per market and plan."""
    A, M, H1 = P.shape
    H = H1 - 1
    mult, share = np.asarray(p["mult"], float), np.asarray(p["share"], float)
    beta, eta = np.asarray(p["beta"], float), np.asarray(p["eta"], float)
    n = np.repeat(n0[None], A, axis=0).astype(np.int64)
    lp, lr = np.log(P), np.log(P / C)
    d = np.maximum(0.0, np.diff(lp, axis=2)) / REF
    shock = p["kappa"] * d ** p["psi"]
    shock[:, :, 1:] += 0.5 * shock[:, :, :-1]                  # the notice lingers into the next bill
    base = (np.asarray(p["alpha"])[None, :, None] + np.asarray(p["u"])[:, None, None] + np.asarray(p["ten"])[None, None, :])[None, :, :, :, None]
    lam = np.asarray(p["lam"], float)
    out = {k: np.zeros((A, H)) for k in ("intents", "saves", "churned", "signups", "customers", "revenue")}
    out["saturated"] = np.zeros((A, H), bool)
    for t in range(1, H1):
        sens = beta[None, None, :] * lp[:, :, t, None] + (p["gamma"] * np.maximum(0.0, lr[:, :, t]) + shock[:, :, t - 1])[:, :, None]   # (A,M,S)
        x = base + sens[:, :, :, None, None] * mult
        if panel is not None:
            at_risk = n[0].sum(-1)
        intents = rng.binomial(n, expit(x))
        tot = intents.sum(axis=(1, 2, 3, 4))
        want = p["save_rate"] * tot
        saved = rng.binomial(intents, p["save_rate"])
        for a in np.flatnonzero(want > p["capacity"]):       # a full desk saves exactly its capacity, spread over the requests
            saved[a] = rng.multivariate_hypergeometric(intents[a].ravel(), int(p["capacity"])).reshape(intents[a].shape)
        n -= intents - saved
        moved = rng.binomial(n[:, :, :, :3], AGE[None, None, None, :, None])
        n[:, :, :, :3] -= moved
        n[:, :, :, 1:] += moved
        new = rng.poisson(lam[None, :, :, None] * share * np.exp(-eta[None, None, :, None] * mult * lr[:, :, t, None, None]))
        n[:, :, :, 0] += new
        i = t - 1
        out["intents"][:, i], out["saves"][:, i] = tot, saved.sum(axis=(1, 2, 3, 4))
        out["churned"][:, i] = tot - out["saves"][:, i]
        out["signups"][:, i] = new.sum(axis=(1, 2, 3))
        per = n.sum(axis=(3, 4))
        out["customers"][:, i] = per.sum(axis=(1, 2))
        out["revenue"][:, i] = (per * LIST_PRICE * P[:, :, t, None]).sum(axis=(1, 2))
        out["saturated"][:, i] = want > p["capacity"]
        if panel is not None:
            panel.append((at_risk, intents[0].sum(-1), saved[0].sum(-1), new[0].sum(-1)))
    out["n"] = n
    return out


def plan_paths(arms, P0, C0, comp, rng, H):
    """Our price and the competitor's for each arm. Random competitor behaviour is drawn once and shared by every arm
    (common random numbers): an exogenous promotion in market 3, month 5 happens in every arm or in none.
    -> P, C (A,M,H+1) and the list of promotions (arm, market, start, months, depth, why)."""
    M, A = len(P0), len(arms)
    starts = rng.random((M, H)) < comp["promo_rate"]
    depth = rng.choice(comp["depths"], (M, H)) if "depths" in comp else rng.uniform(*comp["depth"], (M, H))
    months = rng.choice(comp["durations"], (M, H)) if "durations" in comp else rng.integers(comp["months"][0], comp["months"][1] + 1, (M, H))
    react_u, react_lag = rng.random(M), rng.integers(1, 3, M)
    react_depth = rng.choice(comp["depths"], M) if "depths" in comp else rng.uniform(*comp["depth"], M)
    react_months = rng.choice(comp["durations"], M) if "durations" in comp else rng.integers(comp["months"][0], comp["months"][1] + 1, M)
    q = comp["react_p"]
    P = np.repeat(np.repeat(np.asarray(P0, float)[None, :, None], A, 0), H + 1, 2)
    C = np.repeat(np.repeat(np.asarray(C0, float)[None, :, None], A, 0), H + 1, 2)
    promos = []

    def promo(a, m, s, k, dep, why):
        if s <= H:
            C[a, m, s:s + k] = np.minimum(C[a, m, s:s + k], C0[m] * (1 - dep))
            promos.append((a, m, s, int(k), float(dep), why))

    for a, arm in enumerate(arms):
        P[a, :, arm["month"]:] *= 1 + arm["price_change"]
        if arm["competitor"] == "none":
            continue
        for m, t in np.argwhere(starts):
            promo(a, m, t + 1, months[m, t], depth[m, t], "own promotion")
        if arm["competitor"] == "reactive" and arm["price_change"] >= comp["min_rise"]:
            for m in np.flatnonzero(react_u < q):
                promo(a, m, arm["month"] + react_lag[m], react_months[m], react_depth[m], "reaction to our rise")
        if arm["competitor"] == "discount":
            for m in range(M):
                promo(a, m, arm["discount_start"], arm["discount_months"], arm["discount"], "scenario: competitor discount")
    return P, C, promos


def true_params(b):
    return {**{k: TRUE[k] for k in ("alpha", "beta", "gamma", "kappa", "psi", "eta", "mult", "share")}, "ten": TRUE["tenure"],
            "u": b["u"], "lam": b["lam"], **DESK}


@functools.lru_cache(maxsize=16)
def business(seed, customers=48000):
    """One invented business: market loyalty, price levels, 24 months of warm-up and 36 months of recorded history.
    Management raised prices more often where customers were loyal and after the competitor had raised its own, which is
    the confounding a naive comparison falls into."""
    r = np.random.default_rng(seed)
    M, S, B = len(MARKETS), len(SEGMENTS), len(BANDS)
    u = r.normal(0, TRUE["market_sd"], M)
    w = r.dirichlet(np.full(M, 6.0))
    p0 = np.exp(-0.35 * u + r.normal(0, 0.03, M))                       # loyal markets carry higher prices
    c0 = p0 * np.exp(r.normal(0, 0.03, M))
    base_churn = expit(np.asarray(TRUE["alpha"])[None, :] + u[:, None] - 0.55)
    lam = customers * w[:, None] * SEG_SHARE[None, :] * base_churn * 1.03
    n0 = np.round(customers * w[:, None, None, None] * SEG_SHARE[None, :, None, None] * BAND_SHARE[None, None, :, None] * np.asarray(TRUE["share"])).astype(np.int64)
    b = {"seed": seed, "u": u, "lam": lam}
    p = true_params(b)
    warm = np.repeat(p0[None, :, None], WARMUP + 1, 2), np.repeat(c0[None, :, None], WARMUP + 1, 2)
    # warm-up at constant prices so the tenure and tier mix settle; then the recorded history
    n = simulate(n0, *warm, p, r)["n"][0]
    H = HISTORY_MONTHS
    P, C = np.empty((M, H + 1)), np.empty((M, H + 1))
    P[:, 0], C[:, 0] = p0, c0
    comp_base, promo_left, promo_depth = c0.copy(), np.zeros(M, int), np.zeros(M)
    rises, react_at, raised_at = np.zeros(M, int), {}, np.full(M, -99)
    big = (int(np.argmin(u)), int(r.integers(12, 30)))               # one management test: +10% in the most loyal market
    tc = TRUE_COMPETITOR
    for t in range(1, H + 1):
        for m in range(M):
            if r.random() < tc["drift_p"]:
                comp_base[m] *= 1 + r.uniform(0.03, 0.06)
                raised_at[m] = t
            if promo_left[m] == 0 and t <= H - 4 and (react_at.get(m) == t or r.random() < tc["promo_rate"]):
                promo_left[m], promo_depth[m] = r.integers(tc["months"][0], tc["months"][1] + 1), r.uniform(*tc["depth"])
            C[m, t] = comp_base[m] * (1 - promo_depth[m] if promo_left[m] else 1)
            promo_left[m] = max(0, promo_left[m] - 1)
            rise = 0.10 if (m, t) == big else 0.0
            if not rise and rises[m] < 3 and r.random() < 0.025 + 0.12 * (t - raised_at[m] <= 2) + 0.03 * (u[m] < 0):
                rise = r.uniform(0.02, 0.07)
            P[m, t] = P[m, t - 1] * (1 + rise)
            if rise:
                rises[m] += 1
                if rise >= tc["min_rise"] and r.random() < tc["react_p"]:
                    react_at[m] = t + int(r.integers(1, 3))
    panel = []
    out = simulate(n, P[None], C[None], p, r, panel=panel)
    return {"seed": seed, "u": u, "lam": lam, "P": P, "C": C, "C_obs": C * np.exp(r.normal(0, SCRAPE_SD, C.shape)),
            "at_risk": np.array([x[0] for x in panel]), "intents": np.array([x[1] for x in panel]), "saves": np.array([x[2] for x in panel]),
            "signups": np.array([x[3] for x in panel]), "totals": out, "n_true": out["n"][0], "n_obs": out["n"][0].sum(-1), "P_now": P[:, -1], "C_now": comp_base}
