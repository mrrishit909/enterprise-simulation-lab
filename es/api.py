"""Public API (modular monolith). Blueprint services map to: scenario builder (the business, its history and assumptions),
agent registry (agent types and instances), behavior policy (agents calibrated from history), simulation runtime (seeded
runs and replay), experiment manager (batches of seeded scenarios as a background job), metrics engine (distributions),
sensitivity analysis (Sobol indices and a surrogate) and reporting (decision records with four-eyes sign-off).

    uvicorn es.api:app          python -m core.jobs es.api      # the worker
"""
import hashlib
import json
import time
import uuid
from typing import Literal

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, model_validator

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"scenarios:read", "jobs:read"}
WORK = READ | {"scenarios:write", "agents:calibrate", "runs:create", "experiments:run", "decisions:propose"}
PERMISSIONS = {"viewer": READ, "strategist": WORK, "partner": WORK | {"decisions:approve", "audit:read"}}
app = create_app("enterprise-simulation-lab", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
TRUTH_RUNS = 2000
SCHEME = "numpy default_rng(SeedSequence([entropy, scenario_no])); one generator per scenario, every arm drawn from it"


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def scenario(c, sid):
    s = c.execute("SELECT * FROM scenario WHERE scenario_id = %s", [sid]).fetchone()
    if not s:
        raise Problem(404, "scenario_not_found")
    return s


def observed(c, sid):
    """The calibration dataset as arrays, exactly as world.business recorded it."""
    H, M, S, B = W.HISTORY_MONTHS, len(W.MARKETS), len(W.SEGMENTS), len(W.BANDS)
    obs = {k: np.zeros((H, M, S, B), np.int64) for k in ("at_risk", "intents", "saves")}
    for r in c.execute("SELECT * FROM cohort_month WHERE scenario_id = %s", [sid]):
        i = (r["month"] - 1, r["market"], r["segment"], r["band"])
        obs["at_risk"][i], obs["intents"][i], obs["saves"][i] = r["at_risk"], r["cancel_requests"], r["saves"]
    obs["P"], obs["C_obs"], obs["signups"] = np.zeros((M, H + 1)), np.zeros((M, H + 1)), np.zeros((H, M, S), np.int64)
    for r in c.execute("SELECT * FROM market_month WHERE scenario_id = %s", [sid]):
        obs["P"][r["market"], r["month"]], obs["C_obs"][r["market"], r["month"]] = r["our_price_index"], r["competitor_price_scraped"]
        if r["month"]:
            obs["signups"][r["month"] - 1, r["market"]] = r["signups"]
    if not obs["at_risk"].any():
        raise Problem(409, "no_history")
    obs["n_obs"] = np.array(scenario(c, sid)["state"]["customers"])
    return obs


def policies(c, sid, version=None):
    rows = {r["name"]: r for r in c.execute("SELECT DISTINCT ON (name) * FROM behavior_policy WHERE scenario_id = %s AND (%s::text IS NULL OR version = %s) ORDER BY name, created_at DESC",
                                            [sid, version, version])}
    if len(rows) < 3:
        raise Problem(409, "not_calibrated", "calibrate the agents first: POST /v1/scenarios/{id}/calibrate")
    cal = {"churn": rows["customer_churn"]["params"], "acquisition": rows["customer_acquisition"]["params"], "competitor": rows["competitor_reaction"]["params"]}
    return cal, rows["customer_churn"]["version"]


def state_of(s):
    st = s["state"]
    return {"n": np.array(st["customers"]), "P": np.array(st["price_index"]), "C": np.array(st["competitor_price_index"])}


def money(v):
    return round(float(v))


# ---------------------------------------------------------------- scenario builder and agent registry

class Generate(BaseModel):
    seed: int = Field(18, ge=0, le=10 ** 6)
    customers: int = Field(48000, ge=5000, le=200000)


class ScenarioIn(BaseModel):
    external_ref: str = Field(min_length=1, max_length=60)
    name: str = Field("Subscription business", max_length=120)
    generate: Generate


@app.post("/v1/scenarios", status_code=202, tags=["scenarios"], summary="Build a scenario: the synthetic subscription business, its 36 months of pricing and churn history, its agents and assumptions (background job)")
def create_scenario(body: ScenarioIn, ctx: Ctx = Depends(auth("scenarios:write")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM scenario WHERE external_ref = %s", [body.external_ref]).fetchone():
            raise Problem(409, "scenario_exists")
        return 202, jobs.enqueue(c, ctx, "scenario.generate", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("scenario.generate")
def generate_job(c, job):
    p, t = job["payload"], job["tenant_id"]
    b = W.business(p["generate"]["seed"], p["generate"]["customers"])
    sid = uuid.uuid5(t, f"scenario:{p['external_ref']}")
    st = E.current_state(b)
    c.execute("INSERT INTO scenario (scenario_id, tenant_id, external_ref, name, seed, state, metadata) VALUES (%s,%s,%s,%s,%s,%s,%s)",
              [sid, t, p["external_ref"], p["name"], p["generate"]["seed"], Jsonb({"customers": st["n"].tolist(), "price_index": st["P"].tolist(), "competitor_price_index": st["C"].tolist()}),
               Jsonb({"markets": W.MARKETS, "plans": W.SEGMENTS, "list_price": W.LIST_PRICE.tolist(), "tenure_bands": W.BANDS, "history_months": W.HISTORY_MONTHS})])
    H, M, S, B = b["at_risk"].shape
    db.load(c, "cohort_month", ["scenario_id", "month", "market", "segment", "band", "at_risk", "cancel_requests", "saves", "tenant_id"],
            [(sid, i + 1, m, s, k, int(b["at_risk"][i, m, s, k]), int(b["intents"][i, m, s, k]), int(b["saves"][i, m, s, k]), t)
             for i in range(H) for m in range(M) for s in range(S) for k in range(B)])
    db.load(c, "market_month", ["scenario_id", "month", "market", "our_price_index", "competitor_price_scraped", "signups", "tenant_id"],
            [(sid, i, m, float(b["P"][m, i]), float(b["C_obs"][m, i]), [int(x) for x in b["signups"][i - 1, m]] if i else [0, 0, 0], t) for i in range(H + 1) for m in range(M)])
    types = {"customer_cohort": "Customers of one plan, market and tenure band: cancel or stay each month (probabilistic)",
             "prospect_pool": "Prospects in one market and plan: sign up at a rate that falls as we get dearer than the competitor (probabilistic)",
             "competitor": "The competitor in one market: its own promotions, and a discount in reaction to our rises (rule-based)",
             "retention_desk": "The retention team: saves a share of cancellation requests up to its monthly capacity (rule-based)"}
    tid = {k: uuid.uuid5(sid, f"type:{k}") for k in types}
    for k, d in types.items():
        c.execute("INSERT INTO agent_type (agent_type_id, tenant_id, scenario_id, status, attributes, name) VALUES (%s,%s,%s,'active',%s,%s)", [tid[k], t, sid, Jsonb({"behaviour": d}), k])
    inst = [(f"cohort:{W.MARKETS[m]}:{W.SEGMENTS[s]}:{W.BANDS[k]}", "customer_cohort", {"customers": int(st["n"][m, s, k])}) for m in range(M) for s in range(S) for k in range(B)]
    inst += [(f"prospects:{W.MARKETS[m]}:{W.SEGMENTS[s]}", "prospect_pool", {"signups_last_12_months": int(b["signups"][-12:, m, s].sum())}) for m in range(M) for s in range(S)]
    inst += [(f"competitor:{W.MARKETS[m]}", "competitor", {"price_index_scraped": round(float(st["C"][m]), 4)}) for m in range(M)]
    inst += [("retention_desk", "retention_desk", W.DESK)]
    db.load(c, "agent_instance", ["agent_id", "tenant_id", "scenario_id", "agent_type_id", "key", "state"], [(uuid.uuid5(sid, k), t, sid, tid[ty], k, Jsonb(v)) for k, ty, v in inst])
    for k, v, src in (("retention_desk.save_rate", W.DESK["save_rate"], "operations: share of cancellation requests the desk saves"),
                      ("retention_desk.capacity", W.DESK["capacity"], "operations: saves a month the team can handle"),
                      ("list_price", dict(zip(W.SEGMENTS, W.LIST_PRICE.tolist())), "price book, $ a month"),
                      ("competitor.promotion_threshold", 0.08, "analyst: a scraped price 8% under its recent median is a promotion")):
        c.execute("INSERT INTO assumption (assumption_id, tenant_id, scenario_id, key, value, source) VALUES (%s,%s,%s,%s,%s,%s)", [uuid.uuid5(sid, k), t, sid, k, Jsonb(v), src])
    audit.record(c, worker_ctx(job), "scenario.generated", "scenario", sid, {"customers": int(st["n"].sum()), "agents": len(inst), "history_months": H})
    return {"scenario_id": sid, "customers": int(st["n"].sum()), "agents": len(inst), "cohort_months": H * M * S * B}


@app.get("/v1/scenarios/{sid}", tags=["scenarios"], summary="(+) The business today, its history, its agents and assumptions")
def get_scenario(sid: uuid.UUID, ctx: Ctx = Depends(auth("scenarios:read"))):
    with db.tx(ctx.tenant_id) as c:
        s = scenario(c, sid)
        obs = observed(c, sid)
        agents = c.execute("SELECT t.name, count(*) AS n FROM agent_instance i JOIN agent_type t USING (agent_type_id) WHERE i.scenario_id = %s GROUP BY t.name ORDER BY t.name", [sid]).fetchall()
        assumptions = c.execute("SELECT key, value, source FROM assumption WHERE scenario_id = %s ORDER BY key", [sid]).fetchall()
    n, P = np.array(s["state"]["customers"]), obs["P"]
    by_plan = n.sum(axis=(0, 2))
    mrr = float((n.sum(-1) * W.LIST_PRICE * P[:, -1, None]).sum())
    churned = (obs["intents"] - obs["saves"]).sum(axis=(1, 2, 3))
    rises = np.argwhere(np.diff(np.log(P), axis=1) > 1e-9)
    return jsonable_encoder({
        "scenario_id": sid, "name": s["name"], "external_ref": s["external_ref"], "customers": int(n.sum()), "mrr": money(mrr), "arr": money(12 * mrr),
        "by_plan": [{"plan": p, "customers": int(k), "list_price": float(lp)} for p, k, lp in zip(W.SEGMENTS, by_plan, W.LIST_PRICE)],
        "by_market": [{"market": m, "customers": int(n[i].sum()), "price_index": round(float(P[i, -1]), 3), "competitor_index": round(float(s["state"]["competitor_price_index"][i]), 3)} for i, m in enumerate(W.MARKETS)],
        "history": {"months": W.HISTORY_MONTHS, "churn_rate": [round(float(x), 5) for x in churned / obs["at_risk"].sum(axis=(1, 2, 3))],
                    "save_rate": round(float(obs["saves"].sum() / obs["intents"].sum()), 3), "signups": obs["signups"].sum(axis=(1, 2)).tolist(),
                    "price_rises": [{"market": W.MARKETS[m], "month": int(t) + 1, "change": round(float(P[m, t + 1] / P[m, t] - 1), 3)} for m, t in rises],
                    "our_price": [[round(float(x), 4) for x in row] for row in P], "competitor_scraped": [[round(float(x), 4) for x in row] for row in obs["C_obs"]]},
        "agents": agents, "assumptions": assumptions})


# ---------------------------------------------------------------- behavior policy

class CalibrateIn(BaseModel):
    note: str = Field("", max_length=200)


@app.post("/v1/scenarios/{sid}/calibrate", status_code=201, tags=["agents"], summary="(+) Fit the customers' churn and signup behaviour and the competitor's reaction policy to 36 months of history; store them as versioned behaviour policies")
def calibrate(sid: uuid.UUID, body: CalibrateIn, ctx: Ctx = Depends(auth("agents:calibrate")), idem: str | None = IdemKey):
    def work(c):
        s = scenario(c, sid)
        obs = observed(c, sid)
        cal = E.calibrate(obs)
        ver = f"{E.VERSION}-{int(time.time() * 1000)}"
        ch, co = cal["churn"], cal["competitor"]
        w = np.asarray(ch["weights"])
        psi_90 = [float(np.asarray(ch["psi_grid"])[np.searchsorted(np.cumsum(w), q)]) for q in (0.05, 0.95)]
        pt = E.point(cal)
        for name, ty, params, metrics in (
                ("customer_churn", "customer_cohort", ch, {"method": "binomial hazard, market fixed effects, psi profiled", "cohort_months": int(obs["at_risk"].size), "psi_90": psi_90}),
                ("customer_acquisition", "prospect_pool", cal["acquisition"], {"method": "Poisson regression, market x plan effects"}),
                ("competitor_reaction", "competitor", co, {"method": "episodes read from the scraped price; Beta posterior on the reaction probability"})):
            c.execute("INSERT INTO behavior_policy (policy_id, tenant_id, scenario_id, agent_type, name, version, params, metrics) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                      [uuid.uuid7(), ctx.tenant_id, sid, ty, name, ver, Jsonb(params), Jsonb(metrics)])
        audit.record(c, ctx, "agents.calibrated", "scenario", sid, {"version": ver, "psi": pt["psi"], "reactions": f"{co['reactions']} of {len(co['rises'])}"})
        b = W.business(s["seed"])
        tp = W.true_params(b)
        rises = [round(0.01 * k, 2) for k in range(0, 21)]
        st = E.current_state(obs)
        rng = np.random.default_rng([s["seed"], 3])
        band = np.array([E.rise_response(E.draw(cal, rng)[0], st["n"][..., None], st["P"], st["C"], rises) for _ in range(200)])
        sheet = [(1 + r) ** cal["spreadsheet_elasticity"] - 1 for r in rises]
        a, bb = co["react_a"], co["react_b"]
        rnd = lambda v, d=3: [round(float(x), d) for x in v]      # noqa: E731
        return 201, {"version": ver, "history": {"cohort_months": int(obs["at_risk"].size), "price_rises": len(np.flatnonzero(np.diff(np.log(obs["P"]), axis=1) > 1e-9)), "largest_rise": round(cal["max_rise"], 3),
                                                 "rises_5pct_or_more": len(co["rises"]), "competitor_promotions": co["promotions"], "reactions": co["reactions"]},
                     "churn": {"psi": pt["psi"], "psi_90": psi_90, "psi_profile": [{"psi": p, "weight": round(x, 4)} for p, x in zip(ch["psi_grid"], ch["weights"])],
                               "kappa": round(pt["kappa"], 4), "beta": rnd(pt["beta"]), "gamma": round(pt["gamma"], 3), "market_effects": rnd(pt["u"]), "tenure": rnd(pt["ten"])},
                     "acquisition": {"eta": rnd(pt["eta"])},
                     "competitor": {"reaction_p_mean": round(a / (a + bb), 3), "reaction_p_90": rnd([np.quantile(rng.beta(a, bb, 20000), q) for q in (0.05, 0.95)]),
                                    "promotion_rate": round(co["promo_rate"], 4), "promotion_depths": co["depths"]},
                     "spreadsheet_elasticity": round(cal["spreadsheet_elasticity"], 3),
                     "rise_response": [{"rise": r, "fitted": round(float(np.median(band[:, i])), 4), "lo": round(float(np.quantile(band[:, i], 0.1)), 4), "hi": round(float(np.quantile(band[:, i], 0.9)), 4),
                                        "spreadsheet": round(s_, 4), "truth": round(t_, 4)} for i, (r, s_, t_) in enumerate(zip(rises, sheet, E.rise_response(tp, b["n_true"], b["P_now"], b["C_now"], rises)))],
                     "simulation_truth": {"what": "the world the history came from; the fits never see it", "psi": W.TRUE["psi"], "kappa": W.TRUE["kappa"], "beta": W.TRUE["beta"], "gamma": W.TRUE["gamma"],
                                          "eta": W.TRUE["eta"], "competitor_reaction_p": W.TRUE_COMPETITOR["react_p"], "sensitivity_tiers": {"multiplier": W.TRUE["mult"], "share_of_new_customers": W.TRUE["share"]}},
                     "how": "churn: a binomial discrete-time hazard on cancellation requests per cohort and month, with plan, market and tenure effects, our price, the gap to the competitor's scraped price, and a notice shock kappa*(rise/5%)^psi in the month of a rise (half of it the month after), psi profiled on a grid; signups: Poisson with market x plan effects and the price gap; the competitor: its promotions read off the scraped price, a reaction counted when one starts within two months of a rise of 5% or more"}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- simulation runtime

class Arm(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    price_change: float = Field(0.0, ge=-0.3, le=0.5, description="applied to every market from `month`")
    month: int = Field(1, ge=1, le=24)
    competitor: Literal["reactive", "none", "discount"] = Field("reactive", description="reactive: its own promotions and reactions as calibrated; none: holds its price; discount: the scenario's discount in every market")
    discount: float = Field(0.2, ge=0, le=0.6)
    discount_start: int = Field(2, ge=1, le=24)
    discount_months: int = Field(3, ge=1, le=24)


class Plan(BaseModel):
    arms: list[Arm] = Field(min_length=1, max_length=4)
    horizon_months: int = Field(12, ge=3, le=24)

    @model_validator(mode="after")
    def _months(self):
        if any(a.month > self.horizon_months for a in self.arms):
            raise ValueError("an arm's price change falls after the horizon")
        return self


class RunIn(Plan):
    scenario_id: uuid.UUID
    seed: int = Field(7, ge=0, le=2 ** 62)


def arms_of(plan):
    return [a.model_dump() for a in plan.arms]


def store_run(c, ctx, sid, kind, seed_id, arms, H, version, o, p, comp, promos, scenario_no=0, experiment_id=None, replay_of=None, run_id=None):
    rid = run_id or uuid.uuid7()
    drawn = {"psi": p["psi"], "kappa": round(p["kappa"], 4), "react_p": round(comp["react_p"], 3)}
    c.execute("""INSERT INTO simulation_run (run_id, tenant_id, scenario_id, experiment_id, scenario_no, kind, seed_id, arms, horizon_months, policy_version, revenue, churned, customers, mrr_end, drawn, result_hash, replay_of, created_by)
                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
              [rid, ctx.tenant_id, sid, experiment_id, scenario_no, kind, seed_id, Jsonb(arms), H, version, o["revenue"].sum(1).tolist(), o["churned"].sum(1).tolist(), o["customers"][:, -1].tolist(),
               o["revenue"][:, -1].tolist(), Jsonb(drawn), E.digest(o), replay_of, ctx.actor_id])
    ev = E.events(o, arms, promos, p, comp)
    db.load(c, "event", ["run_id", "seq", "month", "arm", "agent", "kind", "detail", "tenant_id"], [(rid, i, m, a, ag, k, Jsonb(d), ctx.tenant_id) for i, (m, a, ag, k, d) in enumerate(ev)])
    names = ("customers", "churned", "intents", "saves", "signups", "revenue")
    db.load(c, "metric", ["run_id", "arm", "month", "name", "value", "tenant_id"],
            [(rid, a, t + 1, k, float(o[k][a, t]), ctx.tenant_id) for a in range(len(arms)) for t in range(H) for k in names])
    return rid, len(ev)


def summary(o, arms):
    return [{"arm": a["name"], "revenue": money(o["revenue"][i].sum()), "churned": int(o["churned"][i].sum()), "signups": int(o["signups"][i].sum()),
             "customers_end": int(o["customers"][i, -1]), "mrr_end": money(o["revenue"][i, -1]), "desk_saturated_months": int(o["saturated"][i].sum())} for i, a in enumerate(arms)]


@app.post("/v1/runs", status_code=201, tags=["runs"], summary="One seeded run of the calibrated agents through a plan; every event and monthly metric stored")
def create_run(body: RunIn, ctx: Ctx = Depends(auth("runs:create")), idem: str | None = IdemKey):
    def work(c):
        s = scenario(c, body.scenario_id)
        cal, ver = policies(c, body.scenario_id)
        arms = arms_of(body)
        seed_id = uuid.uuid7()
        c.execute("INSERT INTO random_seed (seed_id, tenant_id, entropy, scheme, bit_generator) VALUES (%s,%s,%s,%s,'PCG64')", [seed_id, ctx.tenant_id, body.seed, SCHEME])
        o, p, comp, promos = E.run_one(cal, state_of(s), arms, body.horizon_months, body.seed, 0)
        rid, n = store_run(c, ctx, body.scenario_id, "single", seed_id, arms, body.horizon_months, ver, o, p, comp, promos)
        audit.record(c, ctx, "run.completed", "simulation_run", rid, {"seed": body.seed, "hash": E.digest(o)[:16]})
        return 201, {"run_id": rid, "seed": body.seed, "policy_version": ver, "result_hash": E.digest(o), "events": n, "arms": summary(o, arms)}
    return run(ctx, idem, body, work)


@app.get("/v1/runs/{rid}/events", tags=["runs"], summary="The event stream and monthly metrics of a run")
def run_events(rid: uuid.UUID, arm: int | None = None, ctx: Ctx = Depends(auth("scenarios:read"))):
    with db.tx(ctx.tenant_id) as c:
        r = c.execute("SELECT r.*, s.entropy FROM simulation_run r JOIN random_seed s USING (seed_id) WHERE run_id = %s", [rid]).fetchone()
        if not r:
            raise Problem(404, "run_not_found")
        ev = c.execute("SELECT seq, month, arm, agent, kind, detail FROM event WHERE run_id = %s AND (%s::int IS NULL OR arm IN (-1, %s)) ORDER BY seq", [rid, arm, arm]).fetchall()
        if not ev and r["kind"] == "batch":
            raise Problem(409, "not_recorded", "batch runs keep only their seed and results; replay the run to record its events")
        met = c.execute("SELECT arm, month, name, value FROM metric WHERE run_id = %s ORDER BY arm, month", [rid]).fetchall()
    arms = r["arms"] or []
    series = [{"arm": a["name"], **{k: [m["value"] for m in met if m["arm"] == i and m["name"] == k] for k in ("customers", "churned", "intents", "saves", "signups", "revenue")}} for i, a in enumerate(arms)]
    return jsonable_encoder({"run_id": rid, "kind": r["kind"], "scenario_no": r["scenario_no"], "entropy": r["entropy"], "policy_version": r["policy_version"], "result_hash": r["result_hash"],
                             "replay_of": r["replay_of"], "drawn": r["drawn"], "arms": [a["name"] for a in arms], "events": ev, "series": series})


@app.post("/v1/runs/{rid}/replay", status_code=201, tags=["runs"], summary="Re-run a stored run from its seed and policy version, record every event, and check the result is identical bit for bit")
def replay(rid: uuid.UUID, body: CalibrateIn, ctx: Ctx = Depends(auth("runs:create")), idem: str | None = IdemKey):
    def work(c):
        r = c.execute("SELECT r.*, s.entropy FROM simulation_run r JOIN random_seed s USING (seed_id) WHERE run_id = %s", [rid]).fetchone()
        if not r:
            raise Problem(404, "run_not_found")
        s = scenario(c, r["scenario_id"])
        cal, ver = policies(c, r["scenario_id"], r["policy_version"])
        t0 = time.perf_counter()
        o, p, comp, promos = E.run_one(cal, state_of(s), r["arms"], r["horizon_months"], r["entropy"], r["scenario_no"])
        ms = (time.perf_counter() - t0) * 1000
        new, n = store_run(c, ctx, r["scenario_id"], "replay", r["seed_id"], r["arms"], r["horizon_months"], ver, o, p, comp, promos, r["scenario_no"], r["experiment_id"], rid)
        h = E.digest(o)
        same = h == r["result_hash"] and o["revenue"].sum(1).tolist() == list(r["revenue"]) and o["churned"].sum(1).tolist() == list(r["churned"])
        audit.record(c, ctx, "run.replayed", "simulation_run", new, {"replay_of": str(rid), "identical": same})
        return 201, {"run_id": new, "replay_of": rid, "identical": same, "result_hash": h, "stored_hash": r["result_hash"], "entropy": r["entropy"], "scenario_no": r["scenario_no"],
                     "policy_version": ver, "ms": round(ms, 1), "events": n, "drawn": r["drawn"], "arms": summary(o, r["arms"])}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- experiment manager and metrics engine

class BatchIn(Plan):
    scenario_id: uuid.UUID
    name: str = Field(min_length=1, max_length=120)
    runs: int = Field(10000, ge=100, le=20000)
    seed: int = Field(2026, ge=0, le=2 ** 62)


@app.post("/v1/experiments/batch", status_code=202, tags=["experiments"], summary="Run a batch of seeded scenarios through every arm in a background job; arm 0 is the comparison")
def batch(body: BatchIn, ctx: Ctx = Depends(auth("experiments:run")), idem: str | None = IdemKey):
    def work(c):
        scenario(c, body.scenario_id)
        _, ver = policies(c, body.scenario_id)
        eid = uuid.uuid5(body.scenario_id, f"experiment:{body.name}")
        if c.execute("SELECT 1 FROM experiment WHERE experiment_id = %s", [eid]).fetchone():
            raise Problem(409, "experiment_exists", "experiment names are unique within a scenario")
        seed_id = uuid.uuid7()
        c.execute("INSERT INTO random_seed (seed_id, tenant_id, entropy, scheme, bit_generator) VALUES (%s,%s,%s,%s,'PCG64')", [seed_id, ctx.tenant_id, body.seed, SCHEME])
        c.execute("INSERT INTO experiment (experiment_id, tenant_id, scenario_id, name, arms, horizon_months, runs, seed_id, policy_version, status, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued',%s)",
                  [eid, ctx.tenant_id, body.scenario_id, body.name, Jsonb(arms_of(body)), body.horizon_months, body.runs, seed_id, ver, ctx.actor_id])
        return 202, {**jobs.enqueue(c, ctx, "experiment.batch", {"experiment_id": str(eid)}), "experiment_id": eid}
    return run(ctx, idem, body, work)


def dist(x, bins=30):
    h, edges = np.histogram(x, bins=bins)
    return {"mean": float(x.mean()), "sd": float(x.std()), "quantiles": {str(q): float(np.quantile(x, q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
            "histogram": {"edges": [round(float(e), 2) for e in edges], "counts": h.tolist()}}


@jobs.handler("experiment.batch")
def batch_job(c, job):
    t = job["tenant_id"]
    e = c.execute("SELECT e.*, s.entropy FROM experiment e JOIN random_seed s USING (seed_id) WHERE experiment_id = %s", [job["payload"]["experiment_id"]]).fetchone()
    s = scenario(c, e["scenario_id"])
    cal, ver = policies(c, e["scenario_id"], e["policy_version"])
    arms, H = e["arms"], e["horizon_months"]
    t0 = time.perf_counter()
    res = E.experiment(cal, state_of(s), arms, H, e["runs"], e["entropy"])
    secs = time.perf_counter() - t0
    eid = e["experiment_id"]
    ctx = worker_ctx(job)
    db.load(c, "simulation_run", ["run_id", "tenant_id", "scenario_id", "experiment_id", "scenario_no", "kind", "seed_id", "arms", "horizon_months", "policy_version", "revenue", "churned", "customers", "mrr_end", "drawn", "result_hash", "created_by"],
            [(uuid.uuid5(eid, f"run-{i}"), t, e["scenario_id"], eid, i, "batch", e["seed_id"], Jsonb(arms), H, ver, res["revenue"][i].tolist(), res["churned"][i].tolist(), res["customers"][i].tolist(),
              res["monthly"][i, :, -1].tolist(), Jsonb(res["drawn"][i]), res["hashes"][i], ctx.actor_id) for i in range(e["runs"])])
    rows = []
    for a in range(len(arms)):
        for k, x in (("revenue", res["revenue"][:, a] / 1e6), ("churned", res["churned"][:, a]), ("customers_end", res["customers"][:, a]), ("mrr_end", res["monthly"][:, a, -1] / 1e6)):
            rows.append((a, k, x))
        if a:
            rows += [(a, "revenue_change", (res["revenue"][:, a] - res["revenue"][:, 0]) / 1e6), (a, "churned_change", res["churned"][:, a] - res["churned"][:, 0]),
                     (a, "mrr_end_change", (res["monthly"][:, a, -1] - res["monthly"][:, 0, -1]) / 1e6)]
    db.load(c, "result_distribution", ["experiment_id", "arm", "metric", "mean", "sd", "quantiles", "histogram", "tenant_id"],
            [(eid, a, k, (d := dist(x))["mean"], d["sd"], Jsonb(d["quantiles"]), Jsonb(d["histogram"]), t) for a, k, x in rows])
    # the comparisons: a spreadsheet, and the true world
    obs = observed(c, e["scenario_id"])
    sheet = E.spreadsheet(obs, arms, H, E.spreadsheet_elasticity(obs))
    flat = E.spreadsheet(obs, arms, H, 0.0)
    tr = E.truth(s["seed"], arms, H, TRUTH_RUNS)
    fan = [{"p10": (q := np.quantile(res["monthly"][:, a], [0.1, 0.5, 0.9], axis=0).round())[0].tolist(), "p50": q[1].tolist(), "p90": q[2].tolist(),
            "truth": tr["monthly"][:, a].mean(0).round().tolist()} for a in range(len(arms))]
    metrics = {"seconds": round(secs, 1), "runs": e["runs"], "ms_per_run": round(1000 * secs / e["runs"], 2), "policy_version": ver,
               "spreadsheet": [{"revenue": money(x["revenue"]), "churned": round(x["churned"]), "customers_end": round(x["customers"])} for x in sheet],
               "spreadsheet_flat": [{"revenue": money(x["revenue"]), "churned": round(x["churned"]), "customers_end": round(x["customers"])} for x in flat],
               "spreadsheet_elasticity": round(E.spreadsheet_elasticity(obs), 3),
               "simulation_truth": {"what": f"mean of {TRUTH_RUNS} runs of the true world: true customers, true parameters, the true competitor", "runs": TRUTH_RUNS,
                                    "arms": [{"revenue": money(tr["revenue"][:, a].mean()), "churned": round(float(tr["churned"][:, a].mean())), "customers_end": round(float(tr["customers"][:, a].mean())),
                                              "mrr_end": money(tr["monthly"][:, a, -1].mean()), "revenue_change": money((tr["revenue"][:, a] - tr["revenue"][:, 0]).mean()),
                                              "revenue_change_10_90": [money(np.quantile(tr["revenue"][:, a] - tr["revenue"][:, 0], q)) for q in (0.1, 0.9)],
                                              "mrr_end_change": money((tr["monthly"][:, a, -1] - tr["monthly"][:, 0, -1]).mean()),
                                              "churned_change": round(float((tr["churned"][:, a] - tr["churned"][:, 0]).mean()))} for a in range(len(arms))]},
               "mrr_fan": fan}
    c.execute("UPDATE experiment SET status = 'completed', metrics = %s WHERE experiment_id = %s", [Jsonb(jsonable_encoder(metrics)), eid])
    audit.record(c, ctx, "experiment.completed", "experiment", eid, {"runs": e["runs"], "seconds": round(secs, 1), "policy_version": ver})
    return {"experiment_id": eid, "runs": e["runs"], "seconds": round(secs, 1)}


def experiment_row(c, eid):
    e = c.execute("SELECT e.*, s.entropy FROM experiment e JOIN random_seed s USING (seed_id) WHERE experiment_id = %s", [eid]).fetchone()
    if not e:
        raise Problem(404, "experiment_not_found")
    if e["status"] != "completed":
        raise Problem(409, "experiment_not_completed")
    return e


@app.get("/v1/experiments/{eid}/distributions", tags=["experiments"], summary="Distributions of revenue, churn and customers per arm, the paired change against arm 0, the outliers, a spreadsheet and the truth")
def distributions(eid: uuid.UUID, outliers: int = 5, ctx: Ctx = Depends(auth("scenarios:read"))):
    with db.tx(ctx.tenant_id) as c:
        e = experiment_row(c, eid)
        d = c.execute("SELECT arm, metric, mean, sd, quantiles, histogram FROM result_distribution WHERE experiment_id = %s ORDER BY arm, metric", [eid]).fetchall()
        last = len(e["arms"]) - 1
        worst = c.execute("""SELECT run_id, scenario_no, revenue[%s] - revenue[1] AS revenue_change, churned[%s] - churned[1] AS churned_change, drawn, result_hash
                               FROM simulation_run WHERE experiment_id = %s AND kind = 'batch' ORDER BY 3 LIMIT %s""", [last + 1, last + 1, eid, min(outliers, 20)]).fetchall()
        p_up = [c.execute("SELECT avg((revenue[%s] > revenue[1])::int) AS p, avg((mrr_end[%s] > mrr_end[1])::int) AS m FROM simulation_run WHERE experiment_id = %s AND kind = 'batch'",
                          [a + 1, a + 1, eid]).fetchone() for a in range(1, last + 1)]
        decision = c.execute("SELECT decision_id, result, confidence, approved_by, approved_at, approval_note, final_outcome FROM decision_record WHERE subject_id = %s ORDER BY created_at DESC LIMIT 1", [eid]).fetchone()
    by = {}
    for r in d:
        by.setdefault(r["arm"], {})[r["metric"]] = {k: r[k] for k in ("mean", "sd", "quantiles", "histogram")}
    m = e["metrics"]
    return jsonable_encoder({"experiment_id": eid, "name": e["name"], "runs": e["runs"], "horizon_months": e["horizon_months"], "entropy": e["entropy"], "scheme": SCHEME, "policy_version": e["policy_version"],
                             "seconds": m["seconds"], "ms_per_run": m["ms_per_run"],
                             "arms": [{**a, "index": i, "distributions": by.get(i, {}), "p_revenue_up": float(p_up[i - 1]["p"]) if i else None, "p_mrr_end_up": float(p_up[i - 1]["m"]) if i else None, "mrr_fan": m["mrr_fan"][i],
                                       "spreadsheet": m["spreadsheet"][i], "spreadsheet_flat": m["spreadsheet_flat"][i], "simulation_truth": m["simulation_truth"]["arms"][i]} for i, a in enumerate(e["arms"])],
                             "spreadsheet_elasticity": m["spreadsheet_elasticity"], "truth_runs": m["simulation_truth"]["runs"],
                             "outliers": [{**w, "revenue_change": money(w["revenue_change"]), "churned_change": round(w["churned_change"])} for w in worst],
                             "sensitivity": m.get("sensitivity"), "decision": decision})


# ---------------------------------------------------------------- sensitivity analysis

class SensitivityIn(BaseModel):
    base_samples: int = Field(512, ge=64, le=2048, description="Saltelli base sample N; runs = N x (factors + 2)")
    arm: int = Field(1, ge=1, le=3, description="the arm whose price change is analysed")


@app.post("/v1/experiments/{eid}/sensitivity", status_code=201, tags=["experiments"], summary="(+) Sobol first-order and total indices of the revenue change, and a quadratic surrogate checked against held-out runs")
def sensitivity(eid: uuid.UUID, body: SensitivityIn, ctx: Ctx = Depends(auth("experiments:run")), idem: str | None = IdemKey):
    def work(c):
        e = experiment_row(c, eid)
        if body.arm >= len(e["arms"]):
            raise Problem(422, "no_such_arm")
        s = scenario(c, e["scenario_id"])
        cal, _ = policies(c, e["scenario_id"], e["policy_version"])
        t0 = time.perf_counter()
        out = {**E.sobol(cal, state_of(s), e["arms"][body.arm]["price_change"], e["horizon_months"], body.base_samples, e["entropy"]), "seconds": round(time.perf_counter() - t0, 1), "arm": e["arms"][body.arm]["name"]}
        c.execute("UPDATE experiment SET metrics = metrics || jsonb_build_object('sensitivity', %s::jsonb) WHERE experiment_id = %s", [Jsonb(out), eid])
        top = max(out["factors"], key=lambda f: f["total"])
        audit.record(c, ctx, "experiment.sensitivity", "experiment", eid, {"runs": out["runs"], "top_factor": top["name"], "surrogate_r2": out["surrogate"]["r2_holdout"]})
        return 201, out
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- reporting: decisions

class DecisionIn(BaseModel):
    arm: int = Field(1, ge=1, le=3, description="the proposal; every other non-zero arm is treated as a stress case")
    note: str = Field("", max_length=400)


@app.post("/v1/experiments/{eid}/decision", status_code=201, tags=["decisions"], summary="(+) The policy's recommendation on an arm, with its evidence: adopt, reject, or abstain and send to review")
def propose(eid: uuid.UUID, body: DecisionIn, ctx: Ctx = Depends(auth("decisions:propose")), idem: str | None = IdemKey):
    def work(c):
        e = experiment_row(c, eid)
        if body.arm >= len(e["arms"]):
            raise Problem(422, "no_such_arm")
        if c.execute("SELECT 1 FROM decision_record WHERE subject_id = %s", [eid]).fetchone():
            raise Problem(409, "decision_exists")
        pol = c.execute("SELECT params FROM behavior_policy WHERE scenario_id = %s AND version = %s AND name = 'competitor_reaction'", [e["scenario_id"], e["policy_version"]]).fetchone()["params"]
        pm = {a: c.execute("SELECT avg((revenue[%s] > revenue[1])::int) AS p, avg((mrr_end[%s] > mrr_end[1])::int) AS m FROM simulation_run WHERE experiment_id = %s AND kind = 'batch'",
                           [a + 1, a + 1, eid]).fetchone() for a in range(1, len(e["arms"]))}
        p, pmrr = {a: float(v["p"]) for a, v in pm.items()}, {a: float(v["m"]) for a, v in pm.items()}
        q = {a: c.execute("SELECT quantiles FROM result_distribution WHERE experiment_id = %s AND arm = %s AND metric = 'revenue_change'", [eid, a]).fetchone()["quantiles"] for a in p}
        qm = {a: c.execute("SELECT quantiles FROM result_distribution WHERE experiment_id = %s AND arm = %s AND metric = 'mrr_end_change'", [eid, a]).fetchone()["quantiles"] for a in p}
        rise = e["arms"][body.arm]["price_change"]
        largest = float(np.exp(np.diff(np.log(np.array(observed(c, e["scenario_id"])["P"])), axis=1).max()) - 1)
        extrapolating = rise > largest + 1e-9
        worst = min(p.values())
        sens = e["metrics"].get("sensitivity")
        if p[body.arm] <= 0.2:
            outcome = "reject"
        elif worst >= 0.9 and pmrr[body.arm] >= 0.5 and not extrapolating:
            outcome = "adopt"
        else:
            outcome = "review"
        H = e["horizon_months"]
        reasons = [f"'{e['arms'][a]['name']}': P({H}-month revenue up) {p[a]:.0%}, median {q[a]['0.5']:+.2f} $M; P(month-{H} monthly revenue up) {pmrr[a]:.0%}, median {qm[a]['0.5']:+.3f} $M" for a in p]
        if pmrr[body.arm] < 0.5:
            reasons.append(f"the rise pays over {H} months but by month {H} it earns less each month than the status quo: customers lost outweigh the higher price")
        if extrapolating:
            reasons.append(f"a {rise:.0%} rise is beyond anything in the history (largest {largest:.1%}); the notice-shock curvature is an extrapolation")
        if sens:
            top = max(sens["factors"], key=lambda f: f["total"])
            reasons.append(f"largest source of uncertainty: {top['name']} (total Sobol index {top['total']:.2f})")
        if len(pol["rises"]) < 8:
            reasons.append(f"the competitor's reaction rests on {len(pol['rises'])} episodes")
        result = {"outcome": outcome, "arm": e["arms"][body.arm]["name"], "p_revenue_up": {e["arms"][a]["name"]: round(v, 4) for a, v in p.items()},
                  "p_mrr_end_up": {e["arms"][a]["name"]: round(v, 4) for a, v in pmrr.items()}, "mrr_end_change_median": {e["arms"][a]["name"]: round(qm[a]["0.5"], 4) for a in p},
                  "revenue_change_median": {e["arms"][a]["name"]: round(q[a]["0.5"], 3) for a in p}, "reasons": reasons, "note": body.note}
        state = {"thresholds": {"adopt_if_p_revenue_up_in_every_arm_at_least": 0.9, "and_p_mrr_end_up_at_least": 0.5, "reject_if_p_revenue_up_at_most": 0.2, "abstain_when_extrapolating": True},
                 "extrapolating": extrapolating, "largest_rise_in_history": round(largest, 4), "requires_approval": True}
        did = uuid.uuid7()
        inputs = hashlib.sha256(json.dumps([str(eid), e["policy_version"], e["entropy"], e["runs"]]).encode()).hexdigest()
        c.execute("INSERT INTO decision_record (decision_id, tenant_id, subject_type, subject_id, decision_type, policy_version, inputs_hash, result, confidence, policy_state, created_by) VALUES (%s,%s,'experiment',%s,'pricing',%s,%s,%s,%s,%s,%s)",
                  [did, ctx.tenant_id, eid, e["policy_version"], inputs, Jsonb(result), worst, Jsonb(state), ctx.actor_id])
        audit.record(c, ctx, "decision.proposed", "decision_record", did, {"outcome": outcome, "confidence": round(worst, 4)})
        return 201, {"decision_id": did, **result, "confidence": round(worst, 4), "policy_state": state, "inputs_hash": inputs}
    return run(ctx, idem, body, work)


class ApproveIn(BaseModel):
    final_outcome: Literal["adopt", "reject", "staged_test", "defer"] = Field(description="what the approver decided; may differ from the policy's recommendation")
    note: str = Field(min_length=3, max_length=400)


@app.post("/v1/experiments/{eid}/decision/approve", tags=["decisions"], summary="(+) Sign off the decision; the person who proposed it cannot")
def approve(eid: uuid.UUID, body: ApproveIn, ctx: Ctx = Depends(auth("scenarios:read")), idem: str | None = IdemKey):
    def work(c):
        d = c.execute("SELECT * FROM decision_record WHERE subject_id = %s ORDER BY created_at DESC LIMIT 1", [eid]).fetchone()
        if not d:
            raise Problem(404, "no_decision")
        if d["approved_by"]:
            raise Problem(409, "already_approved")
        if d["created_by"] == ctx.actor_id:
            raise Problem(403, "four_eyes", "the person who proposed a decision cannot approve it")
        if "decisions:approve" not in PERMISSIONS[ctx.role]:
            raise Problem(403, "forbidden", f"role '{ctx.role}' lacks permission 'decisions:approve'")
        c.execute("UPDATE decision_record SET approved_by = %s, approved_at = now(), approval_note = %s, final_outcome = %s WHERE decision_id = %s", [ctx.actor_id, body.note, body.final_outcome, d["decision_id"]])
        c.execute("UPDATE behavior_policy SET approved = true WHERE version = (SELECT policy_version FROM experiment WHERE experiment_id = %s)", [eid])
        audit.record(c, ctx, "decision.approved", "decision_record", d["decision_id"], {"recommended": d["result"]["outcome"], "decided": body.final_outcome, "note": body.note})
        return 200, {"decision_id": d["decision_id"], "recommended": d["result"]["outcome"], "final_outcome": body.final_outcome, "approved_by": ctx.actor_name, "note": body.note, "policy_version": d["policy_version"]}
    return run(ctx, idem, body, work)
