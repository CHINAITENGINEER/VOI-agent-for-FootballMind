"""Shared, versioned observation projection for historical and live text models.

This is a structural input contract, not an information-value heuristic. It
retains unrelated news headlines and never inspects an unread target body.
Legacy artifacts continue to use their original unfiltered projection.
"""
from copy import deepcopy
import re
from urllib.parse import unquote, urlsplit

VERSION = 'observed-news-titles-v1'
LEGACY = 'legacy_all_observed_links'

_CONTROLS = frozenset({
    'read', 'read more', 'learn more', 'view', 'view more', 'more', 'next',
    'previous', 'prev', 'back', 'home', 'news', 'all news', 'latest news',
    'gallery', 'galleries', 'photos', 'video', 'videos', 'mens', "men's",
    'womens', "women's", 'academy', 'fixtures', 'results', 'tickets',
    'shop', 'search', 'menu', 'cookie policy', 'privacy policy',
    'terms & conditions', 'terms and conditions', 'accessibility',
})
_SECTIONS = frozenset({'news', 'article', 'articles', 'story', 'stories'})
_INDEX = frozenset({'all', 'latest', 'index', 'archive', 'archives', 'page', 'category', 'categories', 'tag', 'tags'})
_DATED = re.compile(r'/\d{4}/\d{2}/\d{2}/[^/]+/?$')


def _reason(link):
    title = link.get('title')
    if not isinstance(title, str) or not title.strip():
        return 'empty_observed_label'
    plain = ' '.join(title.split()).casefold().replace('’', "'")
    if plain in _CONTROLS or plain.isdecimal():
        return 'navigation_or_generic_control_label'
    parsed = urlsplit(link.get('url', ''))
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        return 'not_http_article_link'
    path = unquote(parsed.path).casefold()
    parts = [part for part in path.split('/') if part]
    section_article = any(part in _SECTIONS and i + 1 < len(parts)
                          and parts[i + 1] not in _INDEX
                          for i, part in enumerate(parts))
    if not section_article and not _DATED.search(path):
        return 'outside_declared_article_path_projection'
    return None


def project_news_links(links):
    """Keep actual observed article labels; return a reproducible input audit.

    Selection uses URL structure and generic UI labels only. No club, topic,
    match outcome, model gain, or requested candidate changes eligibility.
    Values are preserved byte-for-byte, including label text and time basis.
    """
    kept, excluded, seen = [], [], set()
    for index, link in enumerate(links):
        reason = _reason(link)
        # Identical observation rows must not create extra input evidence.
        key = (link.get('url'), link.get('title'), link.get('parent_url'), link.get('available_at'))
        if reason is None and key in seen:
            reason = 'duplicate_observed_label'
        if reason is not None:
            excluded.append({'index': index, 'url': link.get('url'),
                             'title': link.get('title'), 'reason': reason})
        else:
            seen.add(key)
            kept.append(deepcopy(link))
    return kept, {'schema': VERSION, 'input_labels': len(links),
                  'retained_labels': len(kept), 'excluded_labels': len(excluded),
                  'excluded': excluded,
                  'scope': 'structural_observed_article_labels_not_match_relevance_or_value'}
