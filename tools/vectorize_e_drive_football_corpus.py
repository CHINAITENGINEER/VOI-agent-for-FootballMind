"""Frozen neural text vectors, read-gated views and conservative match links."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from build_e_drive_football_corpus import MODEL_REVISION, digest, write_json, write_rows
from build_e_drive_news_corpus import read_rows

ALIASES = {
    'Man United': ['manchester united', 'man united', 'man utd'],
    'Man City': ['manchester city', 'man city'], 'Tottenham': ['tottenham', 'spurs'],
    'Nott\'m Forest': ['nottingham forest', 'forest'],
    'Wolves': ['wolverhampton wanderers', 'wolves'], 'Newcastle': ['newcastle united', 'newcastle'],
    'Brighton': ['brighton & hove albion', 'brighton and hove albion', 'brighton'],
    'West Ham': ['west ham united', 'west ham'], 'Leicester': ['leicester city', 'leicester'],
    'Sheffield United': ['sheffield united', 'sheff utd'],
    'Ipswich': ['ipswich town', 'ipswich'], 'Leeds': ['leeds united', 'leeds'],
}


def mentions(text, team):
    names = ALIASES.get(team, [team.lower()])
    return any(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', text.lower()) for n in names)


def public_article_view(article):
    """No body, body embeddings, extracted claims or outcomes in unread menu."""
    return {k: article.get(k) for k in ('article_id', 'title', 'url', 'host', 'published_at', 'modified_at',
                                      'source_team', 'source_category', 'source_keywords',
                                      'captured_at', 'provenance', 'historical_version_unknown')}


def link_matches(root):
    matches = read_rows(root / 'structured/matches.jsonl')
    epl = [m for m in matches if m['division'] == 'E0']
    links, coverage = [], []
    for article in read_rows(root / 'news/articles.jsonl'):
        aid = article['article_id']
        try:
            published = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00')).date()
        except (AttributeError, ValueError):
            coverage.append({'article_id': aid, 'status': 'publication_unknown', 'matched': 0})
            continue
        title = article['title']
        metadata_text = title + ' ' + str(article.get('source_category') or '') + ' ' + ' '.join(article.get('source_keywords') or [])
        excluded = re.search(r"\bwomen(?:['’]?s)?\b|\bu-?\d{2}s?\b|\bunder[- ]\d{2}s?\b|\bacademy\b|\bwsl\b|\byouth\b",
                             metadata_text, re.I)
        if excluded:
            coverage.append({'article_id': aid, 'status': 'other_team_category_retained', 'matched': 0})
            continue
        found = 0
        for m in epl:
            days = (datetime.fromisoformat(m['match_date']).date() - published).days
            if not 0 <= days <= 7 or article['source_team'] not in (m['home_team'], m['away_team']):
                continue
            opponent = m['away_team'] if m['home_team'] == article['source_team'] else m['home_team']
            explicit = mentions(title, opponent)
            if not explicit and days > 3:
                continue
            links.append({'article_id': aid, 'match_id': m['match_id'],
                          'binding': 'title_opponent_and_club_date_candidate' if explicit else 'club_temporal_context_candidate',
                          'days_before_match_date': days,
                          'same_day_kickoff_order_verified': False,
                          'historical_version_unknown': True,
                          'requires_semantic_scope_review': True, 'g_training_eligible': False})
            found += 1
        coverage.append({'article_id': aid, 'status': 'candidate_linked' if found else 'no_league_date_match', 'matched': found})
    write_rows(root / 'research/match_article_candidates.jsonl', links)
    write_rows(root / 'research/article_coverage.jsonl', coverage)
    bags = defaultdict(list)
    for link in links:
        bags[link['match_id']].append({k: link[k] for k in ('article_id', 'binding', 'days_before_match_date')})
    write_rows(root / 'research/match_article_bags.jsonl', [
        {'match_id': mid, 'candidates': sorted(items, key=lambda x: x['article_id']),
         'candidate_count': len(items), 'contains_outcome': False,
         'binding_review_required': True, 'prospective_agent_menu': False}
        for mid, items in sorted(bags.items())])
    counts = Counter(x['match_id'] for x in links)
    write_json(root / 'reports/match-links.json', {'candidate_links': len(links), 'candidate_matches': len(counts),
               'matches_with_2plus_candidates': sum(v >= 2 for v in counts.values()),
               'matches_with_3plus_candidates': sum(v >= 3 for v in counts.values()),
               'title_opponent_links': sum(x['binding'].startswith('title_opponent') for x in links),
               'coverage': dict(Counter(x['status'] for x in coverage)),
               'link_validation': 'heuristic_candidates_not_verified_match_facts', 'g_training_rows': 0})


def build_lagged_features(root):
    matches = read_rows(root / 'structured/matches.jsonl')
    labels = {r['match_id']: r for r in read_rows(root / 'structured/labels/results.jsonl')}
    bydate = defaultdict(list)
    for m in matches:
        bydate[m['match_date']].append(m)
    history, rating = defaultdict(list), defaultdict(lambda: 1500.)
    names = ['home_elo', 'away_elo', 'elo_difference', 'home_history_count', 'away_history_count',
             'home_last5_points', 'away_last5_points', 'home_last5_goals_for', 'away_last5_goals_for',
             'home_last5_goals_against', 'away_last5_goals_against']
    rows, values = [], []
    for date in sorted(bydate):
        # All fixtures on the same date are generated before any of that day's labels are applied.
        for m in sorted(bydate[date], key=lambda x: x['match_id']):
            h, a = (m['division'], m['home_team']), (m['division'], m['away_team'])
            hh, ah = history[h][-5:], history[a][-5:]
            sums = lambda seq, col: sum(x[col] for x in seq)
            value = [rating[h], rating[a], rating[h]-rating[a], len(history[h]), len(history[a]),
                     sums(hh, 0), sums(ah, 0), sums(hh, 1), sums(ah, 1), sums(hh, 2), sums(ah, 2)]
            rows.append({'row': len(rows), 'match_id': m['match_id'], 'date': date,
                         'information_cutoff': 'strictly_before_match_date',
                         'split': 'train' if date < '2022-07-01' else ('validation' if date < '2023-07-01' else 'test'),
                         'same_match_outcome_in_features': False})
            values.append(value)
        for m in bydate[date]:
            y = labels.get(m['match_id'])
            if not y:
                continue
            h, a = (m['division'], m['home_team']), (m['division'], m['away_team'])
            score = {'H': 1., 'D': .5, 'A': 0.}[y['outcome']]
            expected = 1/(1+10**((rating[a]-rating[h]-60.)/400.))
            delta = 20*(score-expected)
            rating[h] += delta
            rating[a] -= delta
            history[h].append(({'H': 3, 'D': 1, 'A': 0}[y['outcome']], y['home_goals'], y['away_goals']))
            history[a].append(({'H': 0, 'D': 1, 'A': 3}[y['outcome']], y['away_goals'], y['home_goals']))
    out = root / 'vectors'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'prematch_numeric.npy', np.asarray(values, dtype=np.float32))
    write_rows(out / 'prematch_numeric.rows.jsonl', rows)
    targets = np.asarray([{'H': 0, 'D': 1, 'A': 2}.get(labels.get(r['match_id'], {}).get('outcome'), -1)
                          for r in rows], dtype=np.int8)
    label_dir = root / 'structured/labels'
    np.save(label_dir / 'prematch_numeric_targets.npy', targets)
    write_rows(label_dir / 'prematch_numeric_targets.rows.jsonl',
               [{'row': r['row'], 'match_id': r['match_id']} for r in rows])
    write_json(label_dir / 'prematch_numeric_targets.schema.json',
               {'class_order': ['home', 'draw', 'away'], 'unknown': -1,
                'usage': 'supervised_training_or_evaluation_only',
                'feature_rows': 'vectors/prematch_numeric.rows.jsonl'})
    write_json(out / 'prematch_numeric.schema.json', {'features': names,
               'elo_parameters': {'initial': 1500, 'k': 20, 'home_advantage': 60},
               'parameters_fit': False, 'standardization': 'none_fit_train_partition_only_if_needed',
               'rows': len(rows), 'features_generated': True, 'trained_model': False})


def encode(root, batch_size=32):
    # Set before importing ML packages; model loading is local only and pinned.
    os.environ['HF_HOME'] = str(root / 'cache/huggingface')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TEMP'] = str(root / 'tmp')
    os.environ['TMP'] = str(root / 'tmp')
    (root / 'tmp').mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(4)
    model_path = root / 'models/all-MiniLM-L6-v2' / MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True).eval()
    articles = read_rows(root / 'news/articles.jsonl')
    out = root / 'vectors'
    out.mkdir(parents=True, exist_ok=True)
    title_texts, body_chunks, chunk_rows = [], [], []
    for a in articles:
        title_texts.append(tokenizer.encode(a['title'], add_special_tokens=False)[:254])
        body = Path(a['body_path']).read_text(encoding='utf-8')
        if digest(body.encode('utf-8')) != a['body_sha256']:
            raise ValueError('body digest mismatch')
        ids = tokenizer.encode(body, add_special_tokens=False, truncation=False)
        for start in range(0, len(ids), 222):
            part = ids[start:start+254]
            body_chunks.append(part)
            chunk_rows.append({'row': len(chunk_rows), 'article_id': a['article_id'],
                               'token_start': start, 'token_end': start+len(part),
                               'visibility': 'after_article_read_only', 'body_sha256': a['body_sha256']})
            if start+254 >= len(ids):
                break
    def run(sequences, name):
        output = np.lib.format.open_memmap(out / name, mode='w+', dtype=np.float32, shape=(len(sequences), 384))
        for start in range(0, len(sequences), batch_size):
            subset = sequences[start:start+batch_size]
            prepared = [{'input_ids': [tokenizer.cls_token_id] + seq + [tokenizer.sep_token_id],
                         'attention_mask': [1]*(len(seq)+2)} for seq in subset]
            inputs = tokenizer.pad(prepared, return_tensors='pt')
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1)
                vectors = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            output[start:start+len(subset)] = vectors.numpy()
            if start % (batch_size*8) == 0:
                print('EMBED', name, start, '/', len(sequences), flush=True)
        output.flush()
        return output
    titles = run(title_texts, 'article_titles.npy')
    chunks = run(body_chunks, 'article_body_chunks.npy')
    means = np.zeros((len(articles), 384), dtype=np.float32)
    positions = {a['article_id']: i for i, a in enumerate(articles)}
    for row in chunk_rows:
        means[positions[row['article_id']]] += chunks[row['row']]
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    np.save(out / 'article_body_mean.npy', means)
    write_rows(out / 'article_titles.rows.jsonl', [{'row': i, **public_article_view(a),
               'visibility': 'source_title_only_historical_reconstruction'} for i, a in enumerate(articles)])
    write_rows(out / 'article_body_chunks.rows.jsonl', chunk_rows)
    write_rows(out / 'article_body_mean.rows.jsonl', [{'row': i, 'article_id': a['article_id'],
               'visibility': 'after_article_read_only'} for i, a in enumerate(articles)])
    write_json(root / 'reports/embeddings.json', {'encoder': 'sentence-transformers/all-MiniLM-L6-v2',
               'revision': MODEL_REVISION, 'trained_this_run': False, 'device': 'cpu', 'dimension': 384,
               'articles': len(articles), 'body_chunks': len(body_chunks),
               'body_tokens_not_silently_truncated': True, 'pooling': 'masked_mean_l2',
               'aggregate': 'normalized_sum_of_chunk_vectors',
               'title_vector_finite': bool(np.isfinite(titles).all()),
               'body_vector_finite': bool(np.isfinite(chunks).all()),
               'body_vector_max_unit_norm_error': float(abs(np.linalg.norm(chunks, axis=1)-1).max()),
               'g_unread_body_vectors_allowed': False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['link', 'numeric', 'embed', 'all'], default='all')
    args = p.parse_args()
    if args.stage in ('all', 'link'):
        link_matches(args.root)
    if args.stage in ('all', 'numeric'):
        build_lagged_features(args.root)
    if args.stage in ('all', 'embed'):
        encode(args.root)


if __name__ == '__main__':
    main()
