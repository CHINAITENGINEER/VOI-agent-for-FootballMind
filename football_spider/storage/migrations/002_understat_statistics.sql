-- Additive migration: never rewrite existing raw/fact/feature versions.
ALTER TABLE fm.team_statistics ADD COLUMN ingested_at timestamptz NOT NULL DEFAULT clock_timestamp();
ALTER TABLE fm.team_statistics ADD COLUMN match_version_id uuid REFERENCES fm.match_versions(id);
ALTER TABLE fm.team_statistics ADD COLUMN parser_version text;
ALTER TABLE fm.team_statistics ADD COLUMN evidence jsonb NOT NULL DEFAULT '{}';
ALTER TABLE fm.team_statistics ADD COLUMN source_published_at timestamptz;
CREATE INDEX team_statistics_pit ON fm.team_statistics(team_id,match_id,observed_at,ingested_at);
CREATE UNIQUE INDEX team_statistics_observation_metric ON fm.team_statistics(observation_id,parser_version,match_id,team_id,metric,definition_version);
