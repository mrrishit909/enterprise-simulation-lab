-- Domain model. Names follow the blueprint's tables; additions are marked (+). The blueprint's model_artifact is
-- behavior_policy here: the calibrated parameters of each agent type, versioned.

CREATE TABLE scenario (scenario_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, external_ref text, name text, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(),
  seed int NOT NULL, state jsonb NOT NULL DEFAULT '{}', UNIQUE (tenant_id, external_ref));   -- (+) seed of the synthetic business; state = customers and prices today
SELECT enable_tenant_rls('scenario');

CREATE TABLE agent_type (agent_type_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid REFERENCES scenario, status text NOT NULL, attributes jsonb NOT NULL DEFAULT '{}', observed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(),
  name text NOT NULL, UNIQUE (scenario_id, name));   -- (+) name
SELECT enable_tenant_rls('agent_type');

CREATE TABLE agent_instance (agent_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid NOT NULL REFERENCES scenario, agent_type_id uuid NOT NULL REFERENCES agent_type, key text NOT NULL, state jsonb NOT NULL, UNIQUE (scenario_id, key));
SELECT enable_tenant_rls('agent_instance');

-- (+) The behaviour calibration dataset: what billing and the CRM recorded, month by month.
CREATE TABLE cohort_month (scenario_id uuid NOT NULL REFERENCES scenario, month int NOT NULL, market smallint NOT NULL, segment smallint NOT NULL, band smallint NOT NULL,
  at_risk int NOT NULL CHECK (at_risk >= 0), cancel_requests int NOT NULL CHECK (cancel_requests BETWEEN 0 AND at_risk), saves int NOT NULL CHECK (saves BETWEEN 0 AND cancel_requests), tenant_id uuid NOT NULL REFERENCES tenants,
  PRIMARY KEY (scenario_id, month, market, segment, band));
SELECT enable_tenant_rls('cohort_month');
CREATE TABLE market_month (scenario_id uuid NOT NULL REFERENCES scenario, month int NOT NULL, market smallint NOT NULL, our_price_index double precision NOT NULL CHECK (our_price_index > 0),
  competitor_price_scraped double precision NOT NULL CHECK (competitor_price_scraped > 0), signups int[] NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (scenario_id, month, market));
SELECT enable_tenant_rls('market_month');

CREATE TABLE assumption (assumption_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid NOT NULL REFERENCES scenario, key text NOT NULL, value jsonb NOT NULL, source text NOT NULL, UNIQUE (scenario_id, key));
SELECT enable_tenant_rls('assumption');

CREATE TABLE behavior_policy (policy_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid NOT NULL REFERENCES scenario, agent_type text NOT NULL, name text NOT NULL, version text NOT NULL,
  params jsonb NOT NULL, metrics jsonb NOT NULL DEFAULT '{}', approved boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (scenario_id, name, version));
SELECT enable_tenant_rls('behavior_policy');

CREATE TABLE random_seed (seed_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, entropy bigint NOT NULL, scheme text NOT NULL, bit_generator text NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('random_seed');

CREATE TABLE experiment (experiment_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid NOT NULL REFERENCES scenario, name text NOT NULL, arms jsonb NOT NULL, horizon_months int NOT NULL,
  runs int NOT NULL, seed_id uuid NOT NULL REFERENCES random_seed, policy_version text NOT NULL, status text NOT NULL, metrics jsonb NOT NULL DEFAULT '{}', created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (scenario_id, name));
SELECT enable_tenant_rls('experiment');

CREATE TABLE simulation_run (run_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid NOT NULL REFERENCES scenario, experiment_id uuid REFERENCES experiment, scenario_no int NOT NULL,
  kind text NOT NULL CHECK (kind IN ('single', 'batch', 'replay')), seed_id uuid NOT NULL REFERENCES random_seed, arms jsonb, horizon_months int NOT NULL, policy_version text NOT NULL,
  revenue double precision[] NOT NULL, churned double precision[] NOT NULL, customers double precision[] NOT NULL, mrr_end double precision[] NOT NULL, drawn jsonb NOT NULL, result_hash text NOT NULL, replay_of uuid REFERENCES simulation_run, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('simulation_run');
CREATE UNIQUE INDEX simulation_run_batch ON simulation_run (experiment_id, scenario_no) WHERE kind = 'batch';

CREATE TABLE event (run_id uuid NOT NULL REFERENCES simulation_run, seq int NOT NULL, month int NOT NULL, arm int NOT NULL, agent text NOT NULL, kind text NOT NULL, detail jsonb NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (run_id, seq));
SELECT enable_tenant_rls('event');
CREATE TABLE metric (run_id uuid NOT NULL REFERENCES simulation_run, arm int NOT NULL, month int NOT NULL, name text NOT NULL, value double precision NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (run_id, arm, month, name));
SELECT enable_tenant_rls('metric');

CREATE TABLE result_distribution (experiment_id uuid NOT NULL REFERENCES experiment, arm int NOT NULL, metric text NOT NULL, mean double precision NOT NULL, sd double precision NOT NULL, quantiles jsonb NOT NULL, histogram jsonb NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants,
  PRIMARY KEY (experiment_id, arm, metric));
SELECT enable_tenant_rls('result_distribution');

CREATE TABLE decision_record (decision_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, subject_type text NOT NULL, subject_id uuid NOT NULL, decision_type text NOT NULL, policy_version text NOT NULL,   -- (+) policy_version in place of model_id
  inputs_hash text NOT NULL, result jsonb NOT NULL, confidence double precision, policy_state jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(),
  created_by uuid NOT NULL, approved_by uuid, approved_at timestamptz, approval_note text, final_outcome text);   -- (+) four-eyes sign-off and what the approver decided
SELECT enable_tenant_rls('decision_record');
