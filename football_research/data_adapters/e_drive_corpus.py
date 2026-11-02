"""Read-only access to the external-volume research corpus.

This is a historical data interface, not an Agent execution trace or g label
generator. A title sourced from an article header is not proof of a historical
pre-read menu. Body features require an explicit already-read set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ExternalFootballCorpus:
    def __init__(self, root):
        self.root = Path(root)
        self.articles = {r['article_id']: r for r in self.rows('news/articles.jsonl')}
        self.title_rows = self.rows('vectors/article_titles.rows.jsonl')
        self.body_rows = self.rows('vectors/article_body_mean.rows.jsonl')
        self.positions = {r['article_id']: r['row'] for r in self.title_rows}
        self.title_vectors = np.load(self.root / 'vectors/article_titles.npy', mmap_mode='r')
        self.body_vectors = np.load(self.root / 'vectors/article_body_mean.npy', mmap_mode='r')
        n = len(self.articles)
        if (set(self.positions) != set(self.articles)
                or len(self.title_rows) != n
                or [r['row'] for r in self.title_rows] != list(range(n))
                or [r['row'] for r in self.body_rows] != list(range(n))
                or [r['article_id'] for r in self.title_rows] != [r['article_id'] for r in self.body_rows]
                or self.title_vectors.shape != (n, 384)
                or self.title_vectors.shape != self.body_vectors.shape):
            raise ValueError('corpus vector/record alignment mismatch')

    def rows(self, relative):
        with (self.root / relative).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def unread_candidate(self, article_id):
        record = self.articles[article_id]
        fields = ('article_id', 'title', 'url', 'host', 'source_team', 'source_category', 'source_keywords',
                  'published_at', 'captured_at')
        return {**{k: record.get(k) for k in fields},
                'title_vector': self.title_vectors[self.positions[article_id]].copy(),
                'provenance': 'historical_reconstruction',
                'historical_prequery_title_availability_verified': False}

    def read_evidence(self, article_id, *, already_read_ids):
        if article_id not in already_read_ids:
            raise ValueError('unread article body is not available to the acquisition policy')
        record = self.articles[article_id]
        return {'article_id': article_id,
                'body': Path(record['body_path']).read_text(encoding='utf-8'),
                'body_vector': self.body_vectors[self.positions[article_id]].copy(),
                'provenance': record['provenance']}

    def supervised_targets(self):
        """Explicit training/evaluation-only outcome access, separate from menus."""
        return self.rows('structured/labels/results.jsonl')
