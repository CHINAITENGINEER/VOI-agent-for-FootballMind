"""Exact observed AZ and FC Twente article templates, without extra requests."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from football_agent.inquiry_contract import normalized
from football_agent.link_context import _hidden

VERSION = 'footballmind-live-dutch-registered-body-v1'
BLOCKS = 'p,li,h2,h3,blockquote'
MONTHS = {word: i + 1 for i, word in enumerate(('januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli', 'augustus', 'september', 'oktober', 'november', 'december'))}


def _visible(node):
    return not any(_hidden(p) or p.name in {'nav', 'footer', 'aside', 'form'}
                   for p in [node, *node.parents])


def _one(soup, selector, attr=None):
    values = soup.select(selector)
    if len(values) != 1:
        raise ValueError('unique_' + selector + '_required')
    return values[0].get(attr) if attr else values[0]


def _stamp(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('publication_timezone_missing')
    return result


def _text_blocks(container, roots, extras=()):
    wanted = {id(n) for root in roots for n in root.select(BLOCKS)}
    wanted.update(id(n) for n in extras)
    nodes = [n for n in container.find_all(True) if id(n) in wanted and _visible(n)]
    result = []
    for node in nodes:
        if any(id(parent) in wanted for parent in node.parents):
            continue  # Nested <li><p> or duplicated prose roots are one block.
        clean = BeautifulSoup(str(node), 'html.parser')
        for hidden in list(clean.find_all(True)):
            if hidden.name and (_hidden(hidden) or hidden.name in {'button', 'nav', 'footer', 'aside', 'form', 'svg'}):
                hidden.decompose()
        text = normalized(clean.get_text(' '))
        if text:
            result.append((text, node.name))
    return result


def _az(soup, url, observation):
    main = _one(soup, 'main#main-content')
    heroes = main.select('section.article-image-hero,section.article-video-hero')
    if len(heroes) != 1 or not _visible(heroes[0]):
        raise ValueError('unique_registered_az_article_hero_required')
    hero = heroes[0]
    title = normalized(_one(hero, 'h1').get_text(' '))
    if not title or normalized(_one(soup, 'head title').get_text(' ')) != title:
        raise ValueError('head_title_and_article_heading_mismatch')
    # AZ does not emit canonical/OG/JSON-LD in these actual responses. The
    # observed self form action binds the requested page without deriving
    # body facts from a slug; article page type and DOM/title must also agree.
    forms = soup.select('form[data-analytics-form_id="newsletter"][action]')
    if len(forms) != 1 or urljoin(url, forms[0]['action']) != url:
        raise ValueError('az_explicit_self_form_url_binding_missing')
    for node in soup.select('link[rel="canonical"],meta[property="og:url"]'):
        if urljoin(url, node.get('href') or node.get('content') or '') != url:
            raise ValueError('conflicting_explicit_page_identifier')
    page_types = []
    for script in soup.select('head script:not([src])'):
        for match in re.finditer(r'window\.dataLayer\.push\(\s*(\{.*?\})\s*\);', script.get_text(), re.S):
            try:
                value = json.loads(match.group(1))
            except ValueError:
                continue
            if value.get('event') == 'page_data':
                page_types.append(value.get('page', {}).get('page_type'))
    if len(page_types) != 1 or page_types[0] not in {'articleDetailPage', 'liveblogDetailPage', 'videoDetailPage', 'imageDetailPage'}:
        raise ValueError('registered_az_article_page_type_missing')
    local_raw = normalized(_one(hero, '[aria-label="Publicatiedatum"]').get_text(' '))
    local_date = datetime.strptime(local_raw, '%d.%m.%Y').date().isoformat()
    if local_date > observation.observed_at.date().isoformat():
        raise ValueError('publication_calendar_date_after_observation')
    roots = main.select('.text-block-column-wysiwyg > .content,.article-image-hero-body > .container > .content,.article-video-hero-body > .container > .content')
    extras = []
    if page_types[0] == 'liveblogDetailPage':
        roots += main.select('.liveblog-intro,.liveblog-post .content')
        extras = main.select('.liveblog-post-header > .liveblog-post-bigtitle,.liveblog-post .label-tag')
    roots = [n for n in roots if _visible(n) and not n.find_parent(class_='article-card')]
    return {'title': title, 'blocks': _text_blocks(main, roots, extras), 'published': None,
        'modified': None, 'local_date': local_date, 'publication_raw': local_raw,
        'publication_status': 'date_only_time_zone_unknown', 'authors': [],
        'content_kind': page_types[0], 'clock_conflict': False,
        'selector': 'main#main-content registered wysiwyg/hero-description/liveblog blocks',
        'binding': {'requested_response_url': url, 'explicit_self_form_url': url,
                    'page_type': page_types[0], 'head_and_dom_title': title,
                    'canonical_absent': not bool(soup.select('link[rel="canonical"]'))}}


def _twente(soup, url, observation):
    if _one(soup, 'head link[rel="canonical"]', 'href') != url or _one(soup, 'head meta[property="og:url"]', 'content') != url:
        raise ValueError('canonical_or_og_url_mismatch')
    article = _one(soup, 'main > article')
    if not _visible(article) or 'NewsPost' not in (soup.body.get('class', []) if soup.body else []):
        raise ValueError('registered_twente_newspost_article_required')
    heading = _one(article, 'h1')
    title = normalized(heading.get_text(' '))
    if _hidden(heading) or not title or normalized(_one(soup, 'head meta[property="og:title"]', 'content') or '') != title:
        raise ValueError('article_heading_and_og_title_mismatch')
    # This source's og:type is an explicit empty string; actual article DOM
    # and canonical/title binding identify the observed NewsPost template.
    if _one(soup, 'head meta[property="og:type"]', 'content') not in ('', 'article'):
        raise ValueError('conflicting_og_type')
    published_raw = _one(soup, 'head meta[property="og:datePublished"]', 'content')
    modified_raw = _one(soup, 'head meta[property="og:dateModified"]', 'content')
    published, modified = _stamp(published_raw), _stamp(modified_raw)
    if published > observation.observed_at or modified > observation.observed_at or modified < published:
        raise ValueError('invalid_or_future_source_publication_clock')
    date_text = normalized(_one(article, '[itemprop="datePublished"]').get_text(' '))
    day, month, year = date_text.casefold().split()
    local_date = datetime(int(year), MONTHS[month], int(day)).date().isoformat()
    if local_date > observation.observed_at.date().isoformat():
        raise ValueError('visible_publication_date_after_observation')
    conflict = local_date != published.date().isoformat()
    roots = [n for n in article.select('.prose') if _visible(n)]
    extras = article.select('.app__elements__liveblogpostelement > div:first-child > span') if article.get('id') == 'LiveBlog' else []
    authors = [normalized(n.get('content', '')) for n in soup.select('head meta[name="author"]') if n.get('content')]
    return {'title': title, 'blocks': _text_blocks(article, roots, extras),
        'published': None if conflict else published.isoformat(), 'modified': modified.isoformat(),
        'local_date': local_date, 'publication_raw': date_text,
        'metadata_published_at': published.isoformat(),
        'publication_status': 'source_publication_date_conflict' if conflict else 'declared',
        'authors': authors, 'clock_conflict': conflict,
        'content_kind': 'liveblog_snapshot' if article.get('id') == 'LiveBlog' else 'article',
        'selector': 'main > article .prose semantic blocks and observed liveblog time/title spans',
        'binding': {'canonical_url': url, 'og_url': url, 'heading_text': title,
                    'og_type_empty_per_registered_template': _one(soup, 'head meta[property="og:type"]', 'content') == ''}}


def project_dutch_body(document, raw, observation):
    url = document.get('url', '')
    parts = urlsplit(url)
    if parts.hostname not in {'www.az.nl', 'fctwente.nl'} or (document.get('structured_news') or {}).get('role') == 'article':
        return document

    def reject(reason):
        result = deepcopy(document)
        result['live_dutch_body_projection'] = {'version': VERSION, 'status': 'rejected', 'reason': reason, 'network_requests': 0}
        return result

    if observation.http_status != 200 or observation.error_class or observation.endpoint != url or observation.params or parts.query:
        return reject('unsuccessful_or_unbound_response')
    pattern = r'/inside-az/nieuws/\d{4}/[^/]+/[^/]+/?' if parts.hostname == 'www.az.nl' else r'/nieuws/[^/]+/?'
    if not re.fullmatch(pattern, parts.path):
        return reject('not_a_registered_article_permalink')
    soup = BeautifulSoup(raw, 'html.parser')
    try:
        value = _az(soup, url, observation) if parts.hostname == 'www.az.nl' else _twente(soup, url, observation)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return reject(str(exc))
    blocks = value.pop('blocks')
    text = '\n'.join(t for t, _ in blocks)
    if not blocks or len(text) < 80:
        return reject('no_sufficient_registered_body_text')
    if len(blocks) > 2048 or len(text) > 200000:
        return reject('body_limit_exceeded_no_truncation')
    segments, offset = [], 0
    for index, (body, tag) in enumerate(blocks):
        start = offset + bool(segments)
        segments.append({'text': body, 'start': start, 'end': start + len(body), 'truncated': False,
            'paragraph_index': index, 'content_role': 'heading' if tag in {'h2', 'h3', 'span'} else 'paragraph',
            'content_source': 'registered_same_page_dutch_article_body', 'css_selector': value['selector'],
            'source_url': url, 'source_published_at': value['published'], 'publication_raw': value['publication_raw']})
        offset = start + len(body)
    result = deepcopy(document)
    result.update(kind='article', status='OK', text=text, title=value['title'], text_truncated=False,
        body_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        source_published_at=value['published'], source_modified_at=value['modified'],
        source_publication_local_time=value['local_date'], source_publication_date=value['local_date'],
        publication_status=value['publication_status'], source_authors=value['authors'])
    result['structured_news'] = {'version': VERSION, 'role': 'article', 'headline': value['title'],
        'segments': segments, 'html_body_adapter': VERSION, 'excluded': [],
        'source_language': soup.html.get('lang') if soup.html else None, 'language_source': 'html@lang',
        'body_binding': value['binding'], 'previous_projection': document.get('structured_news')}
    result['live_dutch_body_projection'] = {'version': VERSION, 'status': 'completed',
        'network_requests': 0, 'body_characters': len(text), 'blocks': len(blocks),
        'content_kind': value['content_kind'], 'publication_date_conflict': value['clock_conflict'],
        'source_metadata_published_at': value.get('metadata_published_at'),
        'rendered_visibility_verified': False, 'source_assertions_are_not_verified_match_facts': True,
        'body_scope': value['selector'], 'first_public_available_at': None}
    return result
