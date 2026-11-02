"""Local semantic corpus inspection. Title search is the default unread view."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from build_e_drive_football_corpus import MODEL_REVISION
from build_e_drive_news_corpus import read_rows


def search(root, query, view='title', top_k=5, match_id=None):
    os.environ['HF_HOME'] = str(root / 'cache/huggingface')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(4)
    path = root / 'models/all-MiniLM-L6-v2' / MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, use_safetensors=True).eval()
    inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=256)
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        vector = torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1), dim=1).numpy()[0]
    filename = 'article_titles' if view == 'title' else 'article_body_mean'
    vectors = np.load(root / 'vectors' / (filename+'.npy'), mmap_mode='r')
    rows = read_rows(root / 'vectors' / (filename+'.rows.jsonl'))
    articles = {a['article_id']: a for a in read_rows(root / 'news/articles.jsonl')}
    allowed_ids = set(articles)
    if match_id is not None:
        bags = {r['match_id']: r for r in read_rows(root / 'research/match_article_bags.jsonl')}
        if match_id not in bags:
            raise ValueError('match has no linked candidate bag; global fallback is not permitted')
        allowed_ids = {r['article_id'] for r in bags[match_id]['candidates']}
    scores = vectors @ vector
    result = []
    ranked = [int(i) for i in np.argsort(-scores) if rows[int(i)]['article_id'] in allowed_ids]
    for i in ranked[:top_k]:
        article = articles[rows[int(i)]['article_id']]
        row = {'article_id': article['article_id'], 'similarity': float(scores[i]),
               'title': article['title'], 'source': article['url'], 'published_at': article['published_at'],
               'retrieval_view': view, 'relevance_or_information_value_verified': False}
        row['match_filter'] = match_id
        if view == 'body':
            row['body_path'] = article['body_path']
            row['usage'] = 'offline_research_or_already_read_documents_only'
        result.append(row)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('query')
    p.add_argument('--root', type=Path, default=Path('E:/FootballMind/corpus-v1'))
    p.add_argument('--view', choices=['title', 'body'], default='title')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--match-id', help='Restrict to this existing match candidate bag; no global fallback.')
    args = p.parse_args()
    print(json.dumps(search(args.root, args.query, args.view, args.top_k, args.match_id), ensure_ascii=False, indent=2))
