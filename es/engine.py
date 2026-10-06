"""Calibration of the agents from history, the seeded scenario runner, the spreadsheet baseline, the truth, and global
sensitivity analysis with a surrogate. The calibrated agents never see world.TRUE; only `truth` does, to judge them."""
import functools
import hashlib
import json

import numpy as np
from scipy.special import expit

from . import world as W

VERSION = "es-1"
PSI_GRID = np.round(np.linspace(0.6, 3.0, 25), 2)
NAMES = ([f"alpha_{s}" for s in W.SEGMENTS] + [f"market_{m}" for m in W.MARKETS[1:]] + [f"tenure_{b}" for b in W.BANDS[1:]]
         + [f"beta_{s}" for s in W.SEGMENTS] + ["gamma", "kappa"])


# ---------------------------------------------------------------- calibration

def shock_feature(P, psi):
    d = np.maximum(0.0, np.diff(np.log(P), axis=1)) / W.REF
    f = d ** psi
    f[:, 1:] += 0.5 * f[:, :-1]
    return f                                                     # (M,H): column i is month i+1


def churn_design(obs, psi):
    H, M, S, B = obs["at_risk"].shape
    lp, gap = np.log(obs["P"][:, 1:]), np.maximum(0.0, np.log(obs["P"][:, 1:] / obs["C_obs"][:, 1:]))
    sh = shock_feature(obs["P"], psi)
    t, m, s, b = (a.ravel() for a in np.meshgrid(range(H), range(M), range(S), range(B), indexing="ij"))
    X = np.column_stack([np.eye(S)[s], np.eye(M)[m][:, 1:], np.eye(B)[b][:, 1:], np.eye(S)[s] * lp[m, t][:, None], gap[m, t], sh[m, t]])
    return X, obs["intents"].ravel().astype(float), obs["at_risk"].ravel().astype(float)


def irls(X, y, n=None, iters=50):
    """Logistic (n given: y successes of n) or Poisson (n None) regression by Newton's method. -> coef, covariance, log-likelihood."""
    start = np.log(y + 0.5) if n is None else np.log((y + 0.5) / (n - y + 0.5))
    b = np.linalg.lstsq(X, start, rcond=None)[0]          # least squares on the link scale, then Newton
    for _ in range(iters):
        eta = X @ b
        if n is None:
            mu = np.exp(eta)
            w, g = mu, y - mu
        else:
            p = expit(eta)
            w, g = n * p * (1 - p) + 1e-12, y - n * p
        Hs = X.T @ (w[:, None] * X)
        step = np.linalg.solve(Hs, X.T @ g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    eta = X @ b
    if n is None:
        ll = float((y * eta - np.exp(eta)).sum())
    else:
        ll = float((y * np.log(expit(eta) + 1e-300) + (n - y) * np.log(expit(-eta) + 1e-300)).sum())
    return b, np.linalg.inv(Hs), ll


def fit_churn(obs):
    """Discrete-time hazard on 36 months x 8 markets x 3 plans x 4 tenure bands of cancellation requests. Market fixed
    effects take out the loyalty that drove where prices went up; the notice-shock curvature psi is profiled on a grid."""
    fits = [irls(*churn_design(obs, psi)) for psi in PSI_GRID]
    ll = np.array([f[2] for f in fits])
    w = np.exp(ll - ll.max())
    w /= w.sum()
    return {"psi_grid": PSI_GRID.tolist(), "weights": w.tolist(), "loglik": ll.tolist(), "names": NAMES,
            "theta": [f[0].tolist() for f in fits], "chol": [np.linalg.cholesky(f[1]).tolist() for f in fits]}


def acq_design(obs):
    H, M, S = obs["signups"].shape
    lr = np.log(obs["P"][:, 1:] / obs["C_obs"][:, 1:])
    t, m, s = (a.ravel() for a in np.meshgrid(range(H), range(M), range(S), indexing="ij"))
    return np.column_stack([np.eye(M * S)[m * S + s], -np.eye(S)[s] * lr[m, t][:, None]]), obs["signups"].ravel().astype(float)


def fit_acquisition(obs):
    b, cov, _ = irls(*acq_design(obs))
    return {"theta": b.tolist(), "chol": np.linalg.cholesky(cov).tolist()}


def competitor_episodes(obs, drop=0.08):
    """Promotions read from the scraped price: a fall of more than 8% below the median of the previous three months.
    A promotion starting one or two months after one of our rises of 5% or more counts as a reaction."""
    lc, lp = np.log(obs["C_obs"]), np.log(obs["P"])
    M, H1 = lc.shape
    starts, depths, durations = [], [], []
    for m in range(M):
        t = 1
        while t < H1:
            pre = np.median(lc[m, max(0, t - 3):t])
            if lc[m, t] - pre < -drop:
                k = 1
                while t + k < H1 and lc[m, t + k] - pre < -drop / 2:
                    k += 1
                starts.append((m, t))
                depths.append(float(1 - np.exp(lc[m, t:t + k].mean() - pre)))
                durations.append(k)
                t += k
            else:
                t += 1
    rises = [(m, t) for m in range(M) for t in range(1, H1) if lp[m, t] - lp[m, t - 1] >= np.log(1.05) - 1e-9]
    reacted = [(m, t) for m, t in rises if any(sm == m and t < st <= t + 2 for sm, st in starts)]
    exo = len(starts) - len(reacted)
    return {"rises": [[int(m), int(t)] for m, t in rises], "reactions": len(reacted), "promotions": len(starts),
            "react_a": 1 + len(reacted), "react_b": 1 + len(rises) - len(reacted), "promo_rate": exo / (M * (H1 - 1)), "min_rise": 0.05,
            "depths": [round(d, 4) for d in depths] or [0.15], "durations": durations or [3]}


def spreadsheet_elasticity(obs):
    """The comparison a spreadsheet makes: across markets and months, is churn higher where the price is higher?
    Pooled, no market effects."""
    churned = (obs["intents"] - obs["saves"]).sum(axis=(2, 3))     # (H,M)
    rate = churned / obs["at_risk"].sum(axis=(2, 3))
    x = np.log(obs["P"][:, 1:]).T.ravel()
    return float(np.polyfit(x, np.log(rate.ravel()), 1)[0])


def current_state(obs):
    lc = np.log(obs["C_obs"][:, -6:])
    return {"n": np.asarray(obs["n_obs"]), "P": np.asarray(obs["P"])[:, -1], "C": np.exp(np.median(lc, axis=1))}


def calibrate(obs):
    return {"version": VERSION, "churn": fit_churn(obs), "acquisition": fit_acquisition(obs), "competitor": competitor_episodes(obs),
            "spreadsheet_elasticity": spreadsheet_elasticity(obs), "max_rise": float(np.exp(np.diff(np.log(obs["P"]), axis=1).max()) - 1)}


@functools.lru_cache(maxsize=16)
def calibrate_seed(seed):
    return calibrate(W.business(seed))


def unpack(theta, acq):
    S, M, B = len(W.SEGMENTS), len(W.MARKETS), len(W.BANDS)
    th = np.asarray(theta)
    i = np.cumsum([0, S, M - 1, B - 1, S, 1, 1])
    a = np.asarray(acq)
    return {"alpha": th[i[0]:i[1]], "u": np.r_[0.0, th[i[1]:i[2]]], "ten": np.r_[0.0, th[i[2]:i[3]]], "beta": th[i[3]:i[4]],
            "gamma": float(th[i[4]]), "kappa": float(th[i[5]]), "lam": np.exp(a[:M * S]).reshape(M, S), "eta": a[M * S:],
            "mult": [1.0], "share": [1.0], **W.DESK}


def point(cal):
    """The most likely parameters (no calibration uncertainty)."""
    k = int(np.argmax(cal["churn"]["weights"]))
    return {**unpack(cal["churn"]["theta"][k], cal["acquisition"]["theta"]), "psi": cal["churn"]["psi_grid"][k]}


def draw(cal, rng):
    """One draw of every behaviour parameter from its calibration uncertainty: psi from the profile likelihood, the rest
    from the Laplace approximation at that psi, the competitor's reaction probability from its Beta posterior."""
    ch, ac, co = cal["churn"], cal["acquisition"], cal["competitor"]
    k = rng.choice(len(ch["psi_grid"]), p=np.asarray(ch["weights"]) / np.sum(ch["weights"]))
    th = np.asarray(ch["theta"][k]) + np.asarray(ch["chol"][k]) @ rng.standard_normal(len(ch["theta"][k]))
    a = np.asarray(ac["theta"]) + np.asarray(ac["chol"]) @ rng.standard_normal(len(ac["theta"]))
    p = {**unpack(th, a), "psi": ch["psi_grid"][k]}
    comp = {"react_p": float(rng.beta(co["react_a"], co["react_b"])), "min_rise": co["min_rise"], "promo_rate": co["promo_rate"],
            "depths": co["depths"], "durations": co["durations"]}
    return p, comp


# ---------------------------------------------------------------- scenarios

def digest(o):
    return hashlib.sha256(b"".join(np.ascontiguousarray(o[k], dtype=np.float64).tobytes() for k in ("revenue", "churned", "signups", "customers"))).hexdigest()


def run_one(cal, state, arms, H, entropy, i):
    """Scenario i of an experiment. Its generator is seeded from (entropy, i) alone, so it replays bit for bit on its own."""
    rng = np.random.default_rng([entropy, i])
    p, comp = draw(cal, rng)
    P, C, promos = W.plan_paths(arms, state["P"], state["C"], comp, rng, H)
    o = W.simulate(state["n"][..., None], P, C, p, rng)
    return o, p, comp, promos


def experiment(cal, state, arms, H, runs, entropy):
    A = len(arms)
    rev, churned, cust, monthly = np.zeros((runs, A)), np.zeros((runs, A)), np.zeros((runs, A)), np.zeros((runs, A, H))
    hashes, drawn = [], []
    for i in range(runs):
        o, p, comp, promos = run_one(cal, state, arms, H, entropy, i)
        rev[i], churned[i], cust[i], monthly[i] = o["revenue"].sum(1), o["churned"].sum(1), o["customers"][:, -1], o["revenue"]
        hashes.append(digest(o))
        drawn.append({"psi": p["psi"], "kappa": round(p["kappa"], 4), "react_p": round(comp["react_p"], 3),
                      "reactions": len({m for a, m, *_, why in promos if a == len(arms) - 1 and why.startswith("reaction")})})
    return {"revenue": rev, "churned": churned, "customers": cust, "monthly": monthly, "hashes": hashes, "drawn": drawn}


@functools.lru_cache(maxsize=64)
def _truth(seed, arms_json, H, runs):
    b, arms = W.business(seed), json.loads(arms_json)
    p = W.true_params(b)
    A = len(arms)
    rev, churned, cust, monthly = np.zeros((runs, A)), np.zeros((runs, A)), np.zeros((runs, A)), np.zeros((runs, A, H))
    for i in range(runs):
        rng = np.random.default_rng([seed, 7777, i])
        P, C, _ = W.plan_paths(arms, b["P_now"], b["C_now"], W.TRUE_COMPETITOR, rng, H)
        o = W.simulate(b["n_true"], P, C, p, rng)
        rev[i], churned[i], cust[i], monthly[i] = o["revenue"].sum(1), o["churned"].sum(1), o["customers"][:, -1], o["revenue"]
    return {"revenue": rev, "churned": churned, "customers": cust, "monthly": monthly}


def truth(seed, arms, H, runs=2000):
    """The true world's response: true customers (three hidden sensitivity tiers), true parameters, the true competitor."""
    return _truth(seed, json.dumps(arms, sort_keys=True), H, runs)


def spreadsheet(obs, arms, H, elasticity):
    """Deterministic projection: last year's churn and signups per plan, churn scaled by (price ratio)^elasticity."""
    last = slice(-12, None)
    rate = (obs["intents"][last] - obs["saves"][last]).sum(axis=(0, 1, 3)) / obs["at_risk"][last].sum(axis=(0, 1, 3))   # (S,)
    signups = obs["signups"][last].sum(axis=1).mean(0)                                                                   # (S,)
    st = current_state(obs)
    out = []
    for arm in arms:
        cust = st["n"].sum(-1).astype(float)                                                                            # (M,S)
        rev = churned = 0.0
        for t in range(1, H + 1):
            ratio = (1 + arm["price_change"]) if t >= arm["month"] else 1.0
            ch = cust * rate * ratio ** elasticity
            cust = cust - ch + signups * cust / cust.sum(0)
            churned += ch.sum()
            rev += (cust * W.LIST_PRICE * st["P"][:, None] * ratio).sum()
        out.append({"revenue": rev, "churned": churned, "customers": cust.sum()})
    return out


# ---------------------------------------------------------------- global sensitivity

FACTORS = [("psi", "Notice-shock curvature", 0.8, 2.6), ("kappa", "Notice shock at a 5% rise (x fitted)", 0.5, 1.5),
           ("beta", "Ongoing price sensitivity (x fitted)", 0.5, 1.5), ("gamma", "Sensitivity to the competitor's price (x fitted)", 0.5, 1.5),
           ("eta", "Signup price sensitivity (x fitted)", 0.5, 1.5), ("discount", "Competitor discount depth, 3 months", 0.0, 0.3),
           ("capacity", "Retention desk capacity (x today)", 0.0, 2.0)]


def delta_revenue(cal, state, x, rise, H, rng):
    """12-month revenue of `rise` minus the status quo, at factor values x (one row)."""
    p = point(cal)
    v = dict(zip([f[0] for f in FACTORS], x))
    p = {**p, "psi": v["psi"], "kappa": p["kappa"] * v["kappa"], "beta": p["beta"] * v["beta"], "gamma": p["gamma"] * v["gamma"],
         "eta": p["eta"] * v["eta"], "capacity": W.DESK["capacity"] * v["capacity"]}
    arms = [{"price_change": 0.0, "month": 1, "competitor": "none"},
            {"price_change": rise, "month": 1, "competitor": "discount", "discount": v["discount"], "discount_start": 2, "discount_months": 3}]
    P, C, _ = W.plan_paths(arms, state["P"], state["C"], {"promo_rate": 0.0, "react_p": 0.0, "min_rise": 1.0, "depth": (0, 0), "months": (1, 1)}, rng, H)
    o = W.simulate(state["n"][..., None], P, C, p, rng)
    return (o["revenue"][1].sum() - o["revenue"][0].sum()) / 1e6


def quad(X):
    k = X.shape[1]
    return np.column_stack([np.ones(len(X)), X] + [X[:, i] * X[:, j] for i in range(k) for j in range(i, k)])


def sobol(cal, state, rise=0.12, H=12, N=512, entropy=0):
    """Saltelli sampling with Jansen's estimators: first-order and total indices of each factor for the revenue change.
    Then a quadratic surrogate, fitted on the A sample and scored on the B sample, recomputes the indices from 40,000 points."""
    rng = np.random.default_rng([entropy, 5])
    k = len(FACTORS)
    lo, hi = np.array([f[2] for f in FACTORS]), np.array([f[3] for f in FACTORS])
    A, B = lo + (hi - lo) * rng.random((N, k)), lo + (hi - lo) * rng.random((N, k))
    f = lambda X, tag: np.array([delta_revenue(cal, state, x, rise, H, np.random.default_rng([entropy, tag, j])) for j, x in enumerate(X)])   # noqa: E731
    fA, fB = f(A, 0), f(B, 1)
    fAB = []
    for i in range(k):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB.append(f(ABi, 2 + i))
    var = np.var(np.r_[fA, fB])
    s1 = [float(np.mean(fB * (fAB[i] - fA)) / var) for i in range(k)]
    st = [float(0.5 * np.mean((fA - fAB[i]) ** 2) / var) for i in range(k)]
    z = lambda X: (X - lo) / (hi - lo)       # noqa: E731
    coef = np.linalg.lstsq(quad(z(A)), fA, rcond=None)[0]
    pred = quad(z(B)) @ coef
    r2 = 1 - np.sum((fB - pred) ** 2) / np.sum((fB - fB.mean()) ** 2)
    n2 = 40000
    A2, B2 = rng.random((n2, k)), rng.random((n2, k))
    gA, gB = quad(A2) @ coef, quad(B2) @ coef
    v2 = np.var(np.r_[gA, gB])
    s1s, sts = [], []
    for i in range(k):
        ABi = A2.copy()
        ABi[:, i] = B2[:, i]
        g = quad(ABi) @ coef
        s1s.append(float(np.mean(gB * (g - gA)) / v2))
        sts.append(float(0.5 * np.mean((gA - g) ** 2) / v2))
    return {"factors": [{"key": key, "name": name, "range": [l, h], "first_order": round(a, 3), "total": round(b, 3),
                         "surrogate_first_order": round(c, 3), "surrogate_total": round(d, 3)}
                        for (key, name, l, h), a, b, c, d in zip(FACTORS, s1, st, s1s, sts)],
            "runs": int(N * (k + 2)), "output": f"12-month revenue change from a {rise:.0%} rise, $M", "mean": float(np.mean(fA)), "sd": float(np.std(fA)),
            "p_negative": float(np.mean(np.r_[fA, fB] < 0)), "surrogate": {"kind": "quadratic regression", "terms": int(len(coef)), "r2_holdout": round(float(r2), 4)}}


# ---------------------------------------------------------------- read-outs

def rise_response(p, n, P, C, rises):
    """Cancellation requests in the month of a rise, relative to no rise, across today's customers; the competitor holds its price."""
    mult, beta = np.asarray(p["mult"], float), np.asarray(p["beta"], float)
    lp, lr = np.log(P), np.log(P / C)
    base = (np.asarray(p["alpha"])[None, :, None] + np.asarray(p["u"])[:, None, None] + np.asarray(p["ten"])[None, None, :])[..., None]

    def rate(d):
        sens = beta[None, :] * (lp + d)[:, None] + (p["gamma"] * np.maximum(0.0, lr + d) + p["kappa"] * (d / W.REF) ** p["psi"])[:, None]
        return float((n * expit(base + sens[:, :, None, None] * mult)).sum())

    r0 = rate(0.0)
    return [rate(float(np.log1p(r))) / r0 - 1 for r in rises]


def events(o, arms, promos, p, comp):
    """The event stream of one run: what each agent did, month by month."""
    ev = [(0, -1, "calibration", "parameters_drawn", {"psi": p["psi"], "kappa": round(p["kappa"], 4), "beta": [round(x, 3) for x in p["beta"]],
                                                       "gamma": round(p["gamma"], 3), "eta": [round(x, 3) for x in p["eta"]], "competitor_react_p": round(comp["react_p"], 3)})]
    for a, arm in enumerate(arms):
        if arm["price_change"]:
            ev.append((arm["month"], a, "our pricing", "price_change", {"change": arm["price_change"], "markets": "all"}))
    for a, m, s, k, dep, why in promos:
        ev.append((s, a, "competitor", "promotion", {"market": W.MARKETS[m], "depth": round(dep, 3), "months": k, "why": why}))
    A, H = o["churned"].shape
    for a in range(A):
        for t in range(H):
            if o["saturated"][a, t]:
                ev.append((t + 1, a, "retention desk", "saturated", {"requests": int(o["intents"][a, t]), "saved": int(o["saves"][a, t]), "capacity": W.DESK["capacity"]}))
            if a and o["churned"][a, t] > 1.25 * o["churned"][0, t]:
                ev.append((t + 1, a, "customers", "churn_spike", {"churned": int(o["churned"][a, t]), "status_quo": int(o["churned"][0, t])}))
    return sorted(ev, key=lambda e: (e[0], e[1]))
