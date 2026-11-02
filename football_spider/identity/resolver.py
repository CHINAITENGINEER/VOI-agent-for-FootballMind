from uuid import NAMESPACE_URL, uuid5
from psycopg.types.json import Jsonb
from football_spider.storage.raw_archive import digest


def stable_id(source, kind, source_id):
    return str(uuid5(NAMESPACE_URL, f'footballmind:{source}:{kind}:{source_id}'))


class IdentityResolver:
    def __init__(self, conn):
        self.conn = conn

    def candidate(self, source, kind, sid, candidate, reason, evidence):
        key = digest([source, kind, sid, str(candidate), reason])
        self.conn.execute('''INSERT INTO fm.mapping_candidates(source,entity_type,source_id,candidate_id,reason,evidence,candidate_key)
            VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(candidate_key) DO NOTHING''',
            (source, kind, sid, candidate, reason, Jsonb(evidence), key))

    def resolve(self, source, kind, ref, observed_at):
        sid = ref.source_id
        found = self.conn.execute('''SELECT entity_id FROM fm.entity_source_map
            WHERE source=%s AND entity_type=%s AND source_id=%s AND valid_to IS NULL''', (source, kind, sid)).fetchone()
        if found:
            eid = str(found['entity_id'])
        else:
            eid = stable_id(source, kind, sid)
            self.conn.execute('INSERT INTO fm.entities(id,kind,name,context) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                              (eid, kind, ref.name, Jsonb(ref.model_dump())))
            self.conn.execute('''INSERT INTO fm.entity_source_map(source,entity_type,source_id,entity_id,status,evidence)
                VALUES (%s,%s,%s,%s,%s,%s)''',
                (source, kind, sid, eid, 'source_scoped' if ref.id_quality == 'stable' else 'provisional', Jsonb(ref.model_dump())))
            # Similar or same names are suggestions only, never automatic merges.
            candidates = self.conn.execute('SELECT id,context FROM fm.entities WHERE kind=%s AND lower(name)=lower(%s) AND id<>%s', (kind, ref.name, eid)).fetchall()
            for candidate in candidates:
                self.candidate(source, kind, sid, candidate['id'], 'same_name_requires_context_review', ref.model_dump())
            if ref.id_quality != 'stable':
                self.candidate(source, kind, sid, None, 'no_stable_source_id', ref.model_dump())
        if kind == 'team':
            conflicts = self.conn.execute('SELECT team_id FROM fm.team_aliases WHERE source=%s AND alias=%s AND team_id<>%s', (source, ref.name, eid)).fetchall()
            for conflict in conflicts:
                self.candidate(source, kind, sid, conflict['team_id'], 'alias_conflict', ref.model_dump())
            self.conn.execute('''INSERT INTO fm.team_aliases(team_id,alias,source,context,observed_at)
                VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''', (eid, ref.name, source, Jsonb(ref.model_dump()), observed_at))
        if kind == 'match':
            self.conn.execute('INSERT INTO fm.matches(id) VALUES (%s) ON CONFLICT DO NOTHING', (eid,))
        return eid

    def confirm(self, source, kind, sid, target, reason, allow_context_mismatch=False):
        if not reason.strip():
            raise ValueError('A mapping decision requires evidence/reason')
        with self.conn.transaction():
            entity = self.conn.execute('SELECT * FROM fm.entities WHERE id=%s', (target,)).fetchone()
            if not entity or entity['kind'] != kind:
                raise ValueError('Target entity type mismatch')
            current = self.conn.execute('''SELECT m.*, e.context FROM fm.entity_source_map m JOIN fm.entities e ON e.id=m.entity_id
                WHERE source=%s AND entity_type=%s AND source_id=%s AND valid_to IS NULL FOR UPDATE OF m''', (source, kind, sid)).fetchone()
            if current and str(current['entity_id']) == target and current['status'] == 'confirmed':
                return 'UNCHANGED'
            if current and not allow_context_mismatch:
                for field in ('category', 'country'):
                    left, right = current['context'].get(field), entity['context'].get(field)
                    if left not in (None, 'unknown') and right not in (None, 'unknown') and left != right:
                        raise ValueError(f'Conflicting {field}; explicit reviewed override required')
            if current:
                self.conn.execute('UPDATE fm.entity_source_map SET valid_to=clock_timestamp() WHERE id=%s', (current['id'],))
            self.conn.execute('''INSERT INTO fm.entity_source_map(source,entity_type,source_id,entity_id,status,version,evidence)
                VALUES (%s,%s,%s,%s,'confirmed',%s,%s)''', (source, kind, sid, target, (current['version'] + 1) if current else 1, Jsonb({'reason': reason})))
            self.conn.execute('''INSERT INTO fm.mapping_audit(source,entity_type,source_id,old_id,new_id,reason) VALUES (%s,%s,%s,%s,%s,%s)''',
                              (source, kind, sid, current['entity_id'] if current else None, target, reason))
            self.conn.execute("UPDATE fm.mapping_candidates SET status='reviewed' WHERE source=%s AND entity_type=%s AND source_id=%s", (source, kind, sid))
        return 'CONFIRMED_FUTURE_INGESTIONS_ONLY'
