"""Literal Spanish club news cards; body facts still require an article read."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from football_agent.inquiry_contract import normalized
from football_agent.link_context import _hidden

DIRECTORY_VERSION = 'footballmind-live-ceuta-observed-news-cards-v1'
BODY_VERSION = 'footballmind-live-cadiz-bound-content-blocks-v1'
MONTHS = dict(zip(('enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre').split(), range(1, 13)))


def project_spanish_directory(document, raw, observation):
    """Read one title/date from the exact same visible Ceuta listing card.

    The observed Jet listing uses `Leer más` inside the URL anchor, with the
    actual heading beside it. No slug, alt text, target page or JS is consulted.
    Date-only publication never becomes an invented available_at timestamp.
    """
    url = document.get('url', '')
    parts = urlsplit(url)
    if (parts.hostname != 'adceutafc.com' or parts.path != '/noticias/' or parts.query
            or observation.endpoint != url or observation.params
            or observation.http_status != 200 or observation.error_class):
        return document
    soup = BeautifulSoup(raw, 'html.parser')
    canonical = soup.select('head link[rel="canonical"]')
    if len(canonical) != 1 or canonical[0].get('href') != url:
        return document
    result = deepcopy(document)
    known = {item.get('url'): item for item in result.get('links', [])}
    repaired, rejected = [], []
    for index, card in enumerate(soup.select('.noticia-card.listing-card')):
        if any(_hidden(node) for node in [card, *card.parents]):
            continue
        anchors = card.select('.listing-card-link a.jet-listing-dynamic-link__link[href]')
        headings = card.select('h2.elementor-heading-title')
        if len(anchors) != 1 or len(headings) != 1:
            rejected.append({'card_index': index, 'reason': 'ambiguous_card_url_or_heading'})
            continue
        target = anchors[0].get('href')
        if target not in known or urlsplit(target).hostname != parts.hostname:
            continue
        if any(_hidden(node) for node in [headings[0], *headings[0].parents]):
            continue
        title = normalized(headings[0].get_text(' '))
        if not title:
            continue
        dates = card.select('[itemprop="datePublished"] time')
        publication_raw = normalized(dates[0].get_text(' ')) if len(dates) == 1 else None
        local_date = None
        if publication_raw:
            match = re.fullmatch(r'(\d{1,2}) (\w+) (\d{4})', publication_raw.casefold())
            if match and match[2] in MONTHS:
                try:
                    local_date = datetime(int(match[3]), MONTHS[match[2]], int(match[1])).date().isoformat()
                except ValueError:
                    pass
        known[target].update(title=title[:512], title_source='same_response_registered_news_card_heading',
            title_source_truncated=len(title)>512, source_published_at=None, source_available_at=None,
            publication_raw=publication_raw, source_publication_local_time=local_date,
            publication_status='date_only_time_zone_unknown' if local_date else 'unknown',
            directory_metadata_adapter=DIRECTORY_VERSION,
            metadata_pointer=f'.noticia-card.listing-card[{index}]',
            target_article_fetched_for_context=False)
        repaired.append({'url': target, 'title': title, 'publication_raw': publication_raw,
                         'source_publication_local_time': local_date})
    result['live_spanish_directory_projection'] = {
        'version': DIRECTORY_VERSION, 'status': 'completed', 'cards': len(repaired),
        'records': repaired, 'rejected': rejected, 'network_requests': 0,
        'target_article_requests': 0, 'rendered_visibility_verified': False}
    return result


def project_spanish_body(document, raw, observation):
    """Cádiz's observed ARIA heading + visible rich text + same-page news.

    Read only pageProps.data.news metadata, never authentication or site state.
    Literal CMS text blocks must reproduce the visible registered body exactly.
    """
    url = document.get('url', '')
    parts = urlsplit(url)
    if (parts.hostname != 'www.cadizcf.com' or not re.fullmatch(r'/noticias/[^/]+', parts.path)
            or parts.query or observation.endpoint != url or observation.params
            or observation.http_status != 200 or observation.error_class
            or (document.get('structured_news') or {}).get('role') == 'article'):
        return document

    def reject(reason):
        result = deepcopy(document)
        result['live_spanish_body_projection'] = {'version':BODY_VERSION,'status':'rejected','reason':reason,'network_requests':0}
        return result

    soup = BeautifulSoup(raw, 'html.parser')
    def one(selector):
        nodes = soup.select(selector)
        if len(nodes) != 1:
            raise ValueError('unique_registered_metadata_or_body_required')
        return nodes[0]
    def visible(node):
        return not any(_hidden(p) or p.name in {'nav','footer','aside','form'} for p in [node,*node.parents])
    def paragraphs(roots):
        selected = [node for root in roots for node in root.select('p,li,h2,h3,blockquote') if visible(node)]
        chosen = {id(x) for x in selected}
        values=[]
        for node in selected:
            if any(id(p) in chosen for p in node.parents):
                continue
            clean=BeautifulSoup(str(node),'html.parser')
            for tag in list(clean.find_all(True)):
                if tag.name and (_hidden(tag) or tag.name in {'script','style','nav','footer','aside','button','svg'}):
                    tag.decompose()
            value=normalized(clean.get_text(' '))
            if value:values.append(value)
        return values
    try:
        if (one('head link[rel="canonical"]').get('href') != url
                or one('head meta[property="og:url"]').get('content') != url
                or one('head meta[property="og:type"]').get('content') != 'article'):
            return reject('canonical_or_og_binding_mismatch')
        heading=one('.MkModuleHeaderNews__title[role="heading"][aria-level="1"]')
        if not visible(heading):return reject('hidden_heading')
        title=normalized(heading.get_text(' '))
        if not title or title != normalized(one('head meta[property="og:title"]').get('content','')):
            return reject('heading_and_og_title_mismatch')
        news=json.loads(one('script#__NEXT_DATA__').get_text())['props']['pageProps']['data']['news']
        if (news.get('exclusiveContent') is not False or news.get('slug') != parts.path.split('/')[-1]
                or normalized(news.get('title','')) != title):
            return reject('same_page_public_news_binding_mismatch')
        published=datetime.fromisoformat(str(news.get('publishedAt')).replace('Z','+00:00'))
        if published.tzinfo is None or published > observation.observed_at:
            return reject('invalid_or_future_publication')
        roots=soup.select('.MkGrids__module--content .MkContentNews .MkContentBlocks__richText')
        if not roots or any(not visible(n) for n in roots):
            return reject('registered_visible_news_body_missing')
        values=paragraphs(roots)
        blocks=news.get('contentBlocks')
        if not isinstance(blocks,list) or len(blocks)>256:
            return reject('invalid_content_blocks')
        declared=[]
        for block in blocks:
            if isinstance(block,dict) and block.get('type')=='text' and isinstance(block.get('content'),str):
                declared.extend(paragraphs([BeautifulSoup(block['content'],'html.parser')]))
        if values != declared:
            return reject('visible_and_literal_content_blocks_differ')
        text='\n'.join(values)
        if len(values)<2 or not 160<=len(text)<=200000:
            return reject('insufficient_or_excessive_body_no_truncation')
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return reject('invalid_registered_template_or_metadata')
    segments=[];offset=0
    for index,value in enumerate(values):
        start=offset+bool(segments)
        segments.append({'text':value,'start':start,'end':start+len(value),'truncated':False,
            'paragraph_index':index,'content_role':'paragraph','content_source':'same_page_bound_visible_content_block',
            'css_selector':'.MkGrids__module--content .MkContentNews .MkContentBlocks__richText',
            'source_url':url,'source_published_at':published.isoformat()})
        offset=start+len(value)
    author=news.get('author');names=[]
    if isinstance(author,str) and author.strip():names=[normalized(author)]
    elif isinstance(author,dict) and isinstance(author.get('name'),str):names=[normalized(author['name'])]
    result=deepcopy(document)
    result.update(kind='article',status='OK',title=title,text=text,text_truncated=False,
        source_published_at=published.isoformat(),source_modified_at=None,publication_status='declared',
        source_authors=names,body_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    result['structured_news']={'version':BODY_VERSION,'role':'article','headline':title,'segments':segments,
        'html_body_adapter':BODY_VERSION,'excluded':[],'source_language':soup.html.get('lang') if soup.html else None,
        'language_source':'html@lang','previous_projection':document.get('structured_news'),
        'body_binding':{'canonical_url':url,'og_url':url,'heading_text':title,
            'metadata_pointer':'#__NEXT_DATA__.props.pageProps.data.news',
            'literal_blocks_equal_visible_body':True,'scope':'same_page_body_not_match_truth_or_utility'}}
    result['live_spanish_body_projection']={'version':BODY_VERSION,'status':'completed','body_characters':len(text),
        'paragraphs':len(values),'network_requests':0,'rendered_visibility_verified':False}
    return result
