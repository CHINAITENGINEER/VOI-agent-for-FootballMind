CREATE SCHEMA IF NOT EXISTS fm;
CREATE TABLE fm.raw_observations (
 id uuid PRIMARY KEY, source text NOT NULL, observed_at timestamptz NOT NULL,
 ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(), content_hash text NOT NULL,
 metadata jsonb NOT NULL
);
CREATE INDEX ON fm.raw_observations(source, observed_at);
CREATE TABLE fm.processing (
 observation_id uuid REFERENCES fm.raw_observations(id), parser_version text NOT NULL,
 status text NOT NULL, counts jsonb NOT NULL DEFAULT '{}', error_class text,
 attempts int NOT NULL DEFAULT 1, updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(observation_id, parser_version)
);
CREATE TABLE fm.entities (
 id uuid PRIMARY KEY, kind text NOT NULL CHECK(kind IN ('team','competition','season','player','match')),
 name text NOT NULL, context jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE VIEW fm.teams AS SELECT * FROM fm.entities WHERE kind='team';
CREATE VIEW fm.competitions AS SELECT * FROM fm.entities WHERE kind='competition';
CREATE VIEW fm.seasons AS SELECT * FROM fm.entities WHERE kind='season';
CREATE VIEW fm.players AS SELECT * FROM fm.entities WHERE kind='player';
CREATE TABLE fm.matches (id uuid PRIMARY KEY REFERENCES fm.entities(id));
CREATE TABLE fm.entity_source_map (
 id bigserial PRIMARY KEY, source text NOT NULL, entity_type text NOT NULL, source_id text NOT NULL,
 entity_id uuid NOT NULL REFERENCES fm.entities(id), status text NOT NULL,
 valid_from timestamptz NOT NULL DEFAULT clock_timestamp(), valid_to timestamptz,
 version int NOT NULL DEFAULT 1, evidence jsonb NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX source_map_active ON fm.entity_source_map(source,entity_type,source_id) WHERE valid_to IS NULL;
CREATE INDEX ON fm.entity_source_map(entity_id);
CREATE TABLE fm.team_aliases (
 id bigserial PRIMARY KEY, team_id uuid NOT NULL REFERENCES fm.entities(id), alias text NOT NULL,
 source text NOT NULL, language text NOT NULL DEFAULT 'und', context jsonb NOT NULL DEFAULT '{}',
 observed_at timestamptz NOT NULL, confirmed boolean NOT NULL DEFAULT false,
 UNIQUE(team_id,alias,source)
);
CREATE TABLE fm.mapping_candidates (
 id bigserial PRIMARY KEY, source text NOT NULL, entity_type text NOT NULL, source_id text NOT NULL,
 candidate_id uuid REFERENCES fm.entities(id), reason text NOT NULL, evidence jsonb NOT NULL,
 status text NOT NULL DEFAULT 'pending', created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 candidate_key text NOT NULL UNIQUE
);
CREATE TABLE fm.mapping_audit (
 id bigserial PRIMARY KEY, source text NOT NULL, entity_type text NOT NULL, source_id text NOT NULL,
 old_id uuid, new_id uuid NOT NULL REFERENCES fm.entities(id), reason text NOT NULL,
 changed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE fm.match_versions (
 id uuid PRIMARY KEY, match_id uuid NOT NULL REFERENCES fm.matches(id), source text NOT NULL, source_id text NOT NULL,
 observation_id uuid NOT NULL REFERENCES fm.raw_observations(id), parser_version text NOT NULL,
 schema_version text NOT NULL, home_id uuid NOT NULL REFERENCES fm.entities(id), away_id uuid NOT NULL REFERENCES fm.entities(id),
 competition_id uuid NOT NULL REFERENCES fm.entities(id), season_id uuid REFERENCES fm.entities(id),
 event_time timestamptz NOT NULL, source_published_at timestamptz, first_observed_at timestamptz NOT NULL,
 observed_at timestamptz NOT NULL, ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 valid_from timestamptz NOT NULL, valid_to timestamptz, supersedes_id uuid REFERENCES fm.match_versions(id),
 trusted_snapshot_at timestamptz, content_hash text NOT NULL, status text NOT NULL, record jsonb NOT NULL,
 CHECK(home_id <> away_id), UNIQUE(observation_id,parser_version,source_id)
);
CREATE INDEX ON fm.match_versions(match_id, source, observed_at DESC);
CREATE INDEX ON fm.match_versions(competition_id, event_time);
CREATE INDEX ON fm.match_versions(home_id,event_time);
CREATE INDEX ON fm.match_versions(away_id,event_time);
CREATE VIEW fm.match_version_intervals AS SELECT *,
 lead(valid_from) OVER (PARTITION BY match_id,source ORDER BY observed_at,ingested_at,id) AS effective_valid_to
 FROM fm.match_versions;
CREATE VIEW fm.latest_matches AS SELECT DISTINCT ON (match_id,source) * FROM fm.match_versions
 ORDER BY match_id,source,observed_at DESC,ingested_at DESC,id DESC;
CREATE TABLE fm.odds_history (
 id uuid PRIMARY KEY, match_id uuid NOT NULL REFERENCES fm.matches(id), provider text NOT NULL, bookmaker text,
 market_type text NOT NULL, period text NOT NULL, selection text NOT NULL, line double precision,
 decimal_price double precision NOT NULL CHECK(decimal_price>1 AND decimal_price<'Infinity'::float8),
 source_timestamp timestamptz, observed_at timestamptz NOT NULL, ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 trusted_snapshot_at timestamptz, raw_observation_id uuid NOT NULL REFERENCES fm.raw_observations(id),
 parser_version text NOT NULL, settlement_rule text NOT NULL, quote_key text NOT NULL UNIQUE
);
CREATE INDEX ON fm.odds_history(match_id,observed_at DESC);
CREATE TABLE fm.quality_issues (
 id bigserial PRIMARY KEY, observation_id uuid NOT NULL REFERENCES fm.raw_observations(id),
 rule text NOT NULL, rule_version text NOT NULL, severity text NOT NULL, record_key text NOT NULL,
 issue_key text NOT NULL UNIQUE, detail jsonb NOT NULL, status text NOT NULL DEFAULT 'open',
 created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE fm.quarantined_records (
 observation_id uuid REFERENCES fm.raw_observations(id), parser_version text NOT NULL, record_key text NOT NULL,
 record jsonb NOT NULL, PRIMARY KEY(observation_id,parser_version,record_key)
);
CREATE TABLE fm.quality_resolutions (
 id bigserial PRIMARY KEY, issue_id bigint REFERENCES fm.quality_issues(id), action text NOT NULL,
 note text NOT NULL, resolved_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
-- Ready contracts, not a claim these providers/fields have been connected.
CREATE TABLE fm.team_statistics (
 id uuid PRIMARY KEY, team_id uuid REFERENCES fm.entities(id), match_id uuid REFERENCES fm.matches(id),
 window_start timestamptz, window_end timestamptz, metric text NOT NULL, value double precision,
 unit text NOT NULL, provider text NOT NULL, definition_version text NOT NULL,
 observed_at timestamptz NOT NULL, observation_id uuid REFERENCES fm.raw_observations(id)
);
CREATE TABLE fm.team_state_history (
 id uuid PRIMARY KEY, team_id uuid REFERENCES fm.entities(id), calculation_cutoff timestamptz NOT NULL,
 method_version text NOT NULL, state jsonb NOT NULL, input_manifest_hash text NOT NULL,
 UNIQUE(team_id,calculation_cutoff,method_version,input_manifest_hash)
);
CREATE TABLE fm.player_team_memberships (
 id uuid PRIMARY KEY, player_id uuid REFERENCES fm.entities(id), team_id uuid REFERENCES fm.entities(id),
 valid_from timestamptz NOT NULL, valid_to timestamptz, observed_at timestamptz NOT NULL,
 observation_id uuid REFERENCES fm.raw_observations(id)
);
CREATE TABLE fm.availability_events (
 id uuid PRIMARY KEY, player_id uuid REFERENCES fm.entities(id), team_id uuid REFERENCES fm.entities(id),
 match_id uuid REFERENCES fm.matches(id), status text NOT NULL, event_time timestamptz,
 observed_at timestamptz NOT NULL, observation_id uuid REFERENCES fm.raw_observations(id),
 supersedes_id uuid REFERENCES fm.availability_events(id), evidence jsonb NOT NULL
);
CREATE VIEW fm.injuries AS SELECT * FROM fm.availability_events WHERE status IN ('injured','doubtful','returned');
CREATE TABLE fm.match_events (
 id uuid PRIMARY KEY, match_id uuid REFERENCES fm.matches(id), source text NOT NULL, source_event_id text,
 event_type text NOT NULL, period text NOT NULL, minute int, event_time timestamptz,
 participants jsonb NOT NULL, observed_at timestamptz NOT NULL, observation_id uuid REFERENCES fm.raw_observations(id),
 supersedes_id uuid REFERENCES fm.match_events(id), evidence jsonb NOT NULL
);
CREATE TABLE fm.lineup_versions (
 id uuid PRIMARY KEY, match_id uuid REFERENCES fm.matches(id), team_id uuid REFERENCES fm.entities(id),
 observed_at timestamptz NOT NULL, observation_id uuid REFERENCES fm.raw_observations(id), lineup jsonb NOT NULL
);
CREATE TABLE fm.pipeline_runs (
 id uuid PRIMARY KEY, started_at timestamptz NOT NULL, finished_at timestamptz,
 status text NOT NULL, manifest jsonb NOT NULL
);
CREATE TABLE fm.collector_runs (
 run_id uuid REFERENCES fm.pipeline_runs(id), source text NOT NULL, status text NOT NULL,
 report jsonb NOT NULL, PRIMARY KEY(run_id,source)
);
CREATE TABLE fm.feature_snapshots (
 id uuid PRIMARY KEY, match_id uuid NOT NULL REFERENCES fm.matches(id), prediction_cutoff timestamptz NOT NULL,
 feature_version text NOT NULL, information_mode text NOT NULL CHECK(information_mode IN ('operational','historical_reconstruction')),
 input_manifest_hash text NOT NULL, manifest jsonb NOT NULL, features jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), parquet_path text,
 UNIQUE(match_id,prediction_cutoff,feature_version,information_mode,input_manifest_hash)
);
CREATE TABLE fm.match_labels (
 version_id uuid PRIMARY KEY REFERENCES fm.match_versions(id), match_id uuid REFERENCES fm.matches(id),
 label text NOT NULL CHECK(label IN ('home','draw','away')), period text NOT NULL CHECK(period='regulation'),
 revealed_by timestamptz NOT NULL, time_basis text NOT NULL
);
CREATE TABLE fm.scheduled_jobs (
 job_key text PRIMARY KEY, kind text NOT NULL, due_at timestamptz NOT NULL, status text NOT NULL,
 payload jsonb NOT NULL, attempts int NOT NULL DEFAULT 0, last_run uuid, updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
