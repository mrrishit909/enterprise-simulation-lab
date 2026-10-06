"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, jobs, scenario

from .conftest import bearer

PARTNER, STRAT, VIEWER, OTHER = bearer("atlas", "partner"), bearer("atlas", "strategist"), bearer("atlas", "viewer"), bearer("other", "partner")
NOBODY = "00000000-0000-0000-0000-000000000000"
ARMS = [{"name": "Status quo"}, {"name": "Rise", "price_change": 0.05}]


def test_nothing_runs_before_a_scenario_and_its_agents(client):
    assert client.post("/v1/runs", headers=STRAT, json={"scenario_id": NOBODY, "arms": ARMS}).json()["code"] == "scenario_not_found"
    assert client.post("/v1/scenarios", headers=VIEWER, json={"external_ref": "X", "generate": {"seed": 1}}).status_code == 403
    assert client.post("/v1/runs", headers=STRAT, json={"scenario_id": NOBODY, "arms": [{"name": "late", "month": 13}]}).status_code == 422


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    assert demo["gen"]["result"]["agents"] == 129 and one("SELECT count(*) AS n FROM cohort_month")["n"] == 3456
    assert demo["biz"]["customers"] > 50000 and len(demo["biz"]["history"]["price_rises"]) > 10
    cal = demo["cal"]
    assert one("SELECT count(*) AS n FROM behavior_policy")["n"] == 3 and cal["spreadsheet_elasticity"] < 0
    r4, r12 = cal["rise_response"][4], cal["rise_response"][12]
    assert abs(r4["fitted"] / r4["truth"] - 1) < 0.15 and abs(r12["fitted"] / r12["truth"] - 1) < 0.5 and r12["spreadsheet"] < 0
    assert r12["hi"] - r12["lo"] > 3 * (r4["hi"] - r4["lo"])        # close inside the history; beyond it, an extrapolation that knows it
    assert demo["one"]["events"] > 5 and any(e["kind"] == "promotion" for e in demo["oneev"]["events"])
    d = demo["dist"]
    assert d["runs"] == 10000 and one("SELECT count(*) AS n FROM simulation_run WHERE kind = 'batch'")["n"] == 10000
    rise, stress = d["arms"][1], d["arms"][2]
    assert rise["p_revenue_up"] > 0.9 and rise["distributions"]["revenue_change"]["mean"] > stress["distributions"]["revenue_change"]["mean"]
    q = rise["distributions"]["revenue_change"]["quantiles"]
    assert q["0.01"] < rise["simulation_truth"]["revenue_change"] / 1e6 < q["0.99"]
    assert rise["spreadsheet"]["revenue"] - d["arms"][0]["spreadsheet"]["revenue"] > 2 * rise["simulation_truth"]["revenue_change"]
    rp = demo["replay"]
    assert rp["identical"] and rp["result_hash"] == rp["stored_hash"] == d["outliers"][0]["result_hash"] and rp["replay_of"] == d["outliers"][0]["run_id"]
    assert 0 < len(demo["rev"]["events"]) < rp["events"] and demo["rev"]["series"][0]["revenue"]
    s = demo["sens"]
    assert len(s["factors"]) == 7 and s["surrogate"]["r2_holdout"] > 0.9 and all(-0.1 < f["first_order"] <= f["total"] + 0.1 for f in s["factors"])
    p = demo["prop"]
    assert p["outcome"] == "review" and p["policy_state"]["extrapolating"] and demo["self"]["code"] == "four_eyes"
    assert demo["approve"]["final_outcome"] == "staged_test" and one("SELECT count(*) AS n FROM behavior_policy WHERE approved")["n"] == 3
    assert demo["audit"]["chain_valid"] and {"scenario.generated", "agents.calibrated", "run.completed", "experiment.completed", "run.replayed", "experiment.sensitivity",
                                             "decision.proposed", "decision.approved"} <= {e["action"] for e in demo["audit"]["events"]}


def test_validation_replay_and_approval_rules(client, demo):
    sid, eid = demo["gen"]["result"]["scenario_id"], demo["batch"]["result"]["experiment_id"]
    assert client.post("/v1/experiments/batch", headers=STRAT, json={"scenario_id": sid, "name": "Price +12% for 2027", "arms": ARMS, "runs": 100}).json()["code"] == "experiment_exists"
    assert client.post("/v1/experiments/batch", headers=STRAT, json={"scenario_id": sid, "name": "big", "arms": ARMS, "runs": 50000}).status_code == 422
    assert client.get(f"/v1/runs/{demo['dist']['outliers'][1]['run_id']}/events", headers=VIEWER).json()["code"] == "not_recorded"
    assert client.post(f"/v1/experiments/{eid}/decision/approve", headers=PARTNER, json={"final_outcome": "adopt", "note": "again"}).json()["code"] == "already_approved"
    assert client.post(f"/v1/experiments/{eid}/decision", headers=STRAT, json={"arm": 1}).json()["code"] == "decision_exists"
    h = {**STRAT, "Idempotency-Key": "run-replay"}
    body = {"scenario_id": sid, "arms": ARMS, "seed": 11, "horizon_months": 6}
    a, b = client.post("/v1/runs", headers=h, json=body), client.post("/v1/runs", headers=h, json=body)
    assert a.status_code == 201 and a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true"
    assert client.post("/v1/runs", headers=h, json={**body, "seed": 12}).json()["code"] == "idempotency_key_reused"
    again = client.post("/v1/runs", headers=STRAT, json={**body, "seed": 11})            # a new run with the same seed is the same run
    assert again.json()["result_hash"] == a.json()["result_hash"] and again.json()["run_id"] != a.json()["run_id"]


def test_tenant_isolation_and_append_only_audit(client, demo):
    sid, eid, rid = demo["gen"]["result"]["scenario_id"], demo["batch"]["result"]["experiment_id"], demo["one"]["run_id"]
    assert client.get(f"/v1/scenarios/{sid}", headers=OTHER).status_code == 404
    assert client.get(f"/v1/experiments/{eid}/distributions", headers=OTHER).status_code == 404
    assert client.get(f"/v1/runs/{rid}/events", headers=OTHER).status_code == 404
    assert client.post(f"/v1/runs/{rid}/replay", headers=OTHER, json={}).status_code == 404
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Ridgeline%%'").fetchone()["tenant_id"]
    assert one("SELECT count(*) AS n FROM simulation_run", tenant=other)["n"] == 0 and one("SELECT count(*) AS n FROM cohort_month", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
