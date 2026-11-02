"""Validate finished data/vector alignment and produce a readable delivery report."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_e_drive_football_corpus import digest, write_json, write_rows
from build_e_drive_news_corpus import read_rows
from football_research.data_adapters.e_drive_corpus import ExternalFootballCorpus


def validate(root):
    articles = read_rows(root / 'news/articles.jsonl')
    matches = read_rows(root / 'structured/matches.jsonl')
    labels = read_rows(root / 'structured/labels/results.jsonl')
    links = read_rows(root / 'research/match_article_candidates.jsonl')
    ids = {a['article_id'] for a in articles}
    match_ids = {m['match_id'] for m in matches}
    problems = []
    if len(ids) != len(articles) or len(match_ids) != len(matches):
        problems.append('duplicate identities')
    for a in articles:
        body = Path(a['body_path']).read_bytes()
        if digest(body) != a['body_sha256']:
            problems.append('body hash mismatch ' + a['article_id'])
    if any(l['article_id'] not in ids or l['match_id'] not in match_ids for l in links):
        problems.append('unresolved match/article reference')
    vector_files = ('article_titles', 'article_body_chunks', 'article_body_mean', 'prematch_numeric')
    shapes = {}
    for name in vector_files:
        x = np.load(root / 'vectors' / (name + '.npy'), mmap_mode='r')
        rows = read_rows(root / 'vectors' / (name + '.rows.jsonl'))
        shapes[name] = list(x.shape)
        if len(x) != len(rows) or not np.isfinite(x).all():
            problems.append('vector alignment/nonfinite ' + name)
        if name.startswith('article') and not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
            problems.append('vector normalization ' + name)
    corpus = ExternalFootballCorpus(root)
    chunk_rows = read_rows(root / 'vectors/article_body_chunks.rows.jsonl')
    max_chunk_tokens = max((r['token_end'] - r['token_start'] for r in chunk_rows), default=0)
    if max_chunk_tokens > 254:
        problems.append('body chunk exceeds encoder content budget')
    target_rows = read_rows(root / 'structured/labels/prematch_numeric_targets.rows.jsonl')
    feature_rows = read_rows(root / 'vectors/prematch_numeric.rows.jsonl')
    targets = np.load(root / 'structured/labels/prematch_numeric_targets.npy')
    if ([r['match_id'] for r in target_rows] != [r['match_id'] for r in feature_rows]
            or len(targets) != len(feature_rows) or not np.isin(targets, [-1, 0, 1, 2]).all()):
        problems.append('supervised target alignment')
    if articles:
        aid = articles[0]['article_id']
        try:
            corpus.read_evidence(aid, already_read_ids=set())
            problems.append('unread body incorrectly exposed')
        except ValueError:
            pass
        assert corpus.read_evidence(aid, already_read_ids={aid})['body']
    body_groups = {}
    for a in articles:
        body_groups.setdefault(a['body_sha256'], []).append(a['article_id'])
    write_rows(root / 'research/duplicate_body_groups.jsonl', [{'body_sha256': sha, 'article_ids': group}
               for sha, group in body_groups.items() if len(group) > 1])
    byte_count = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    report = {'status': 'PASS' if not problems else 'FAIL', 'errors': problems, 'matches': len(matches),
              'settled_labels': len(labels), 'unique_team_names': len({m[k] for m in matches for k in ('home_team', 'away_team')}),
              'league_match_counts': dict(Counter(m['division'] for m in matches)),
              'article_bodies': len(articles), 'unique_body_hashes': len(body_groups),
              'news_hosts': dict(Counter(a['host'] for a in articles)),
              'news_publication_years': dict(sorted(Counter(str(a['published_at'])[:4] for a in articles).items())),
              'match_links': json.loads((root / 'reports/match-links.json').read_text(encoding='utf-8')),
              'vectors': shapes, 'data_bytes_at_validation': byte_count,
              'max_body_content_tokens_per_chunk': max_chunk_tokens,
              'max_encoder_tokens_including_special_tokens': max_chunk_tokens + 2,
              'decimal_gb_at_validation': byte_count / 1e9, 'binary_gib_at_validation': byte_count / 2**30,
              'new_f_or_g_trained': False, 'new_g_gain_labels': 0,
              'historical_corpus_not_prospective_trajectories': True,
              'basis': 'source_asserted_dates_currently_retrieved_versions',
              'next_modeling_step': 'review_match_bindings_then_freeze_f_and_generate_read_transitions'}
    write_json(root / 'reports/delivery.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('E:/FootballMind/corpus-v1'))
    args = p.parse_args()
    report = validate(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
