"""Public-source tools for the human trial; no changes to frozen research tools."""
from __future__ import annotations

from datetime import datetime, timezone
import ipaddress
import json
import socket
import time
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from uuid import uuid4
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests

from .inquiry_transport import request_header_profile, SPFTLL_HEADERS
from .transport_probe import DiagnosticSession, exception_facts
from .structured_sources import registered_request, project_platform

VERSION = 'experience-public-sources-v1'
SEARCH_ENGINES = {
    'bing': 'https://www.bing.com/search',
    'duckduckgo': 'https://html.duckduckgo.com/html/',
    'news': 'https://news.google.com/rss/search',
}


def clean_url(value):
    if not isinstance(value, str) or len(value) > 2500:
        raise ValueError('URL_FORMAT')
    p = urlsplit(value.strip())
    if p.scheme != 'https' or not p.hostname or p.username or p.password or p.port not in (None, 443):
        raise ValueError('PUBLIC_HTTPS_REQUIRED')
    hostname = p.hostname.lower()
    if hostname in {'localhost'} or hostname.endswith(('.local', '.internal', '.localhost')):
        raise ValueError('PUBLIC_HOST_REQUIRED')
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not ip.is_global:
            raise ValueError('PUBLIC_HOST_REQUIRED')
    for key, _ in parse_qsl(p.query, keep_blank_values=True):
        if key.lower() in {'token', 'key', 'api_key', 'authorization', 'password', 'secret', 'access_token'}:
            raise ValueError('CREDENTIAL_URL_REJECTED')
    return urlunsplit(('https', p.netloc.lower(), p.path or '/', p.query, ''))


def check_public_dns(url):
    hostname = urlsplit(clean_url(url)).hostname
    addresses = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(r[4][0]).is_global for r in addresses):
        raise ValueError('PUBLIC_DNS_REQUIRED')


def compact(text):
    return ' '.join(text.split())


def project_body(body, url, status, content_type='', error=None):
    """Technical descriptors are not semantic relevance or truth labels."""
    doc = {'url': url, 'title': '', 'text': '', 'links': [], 'status': 'unavailable',
           'http_status': status, 'error': error, 'content_kind': 'unknown',
           'text_truncated': False, 'signals': [], 'semantic_relevance': 'unassessed'}
    if error or status is None:
        doc['signals'] = ['request_failed']
        return doc
    if status in {301, 302, 303, 307, 308}:
        doc['signals'] = ['redirect_not_completed']
        return doc
    if status >= 400:
        doc['signals'] = ['rate_limited' if status == 429 else 'access_blocked' if status in {401,403} else 'http_error']
        return doc
    if body.startswith(b'%PDF') or content_type.lower().startswith(('image/','audio/','video/','application/octet-stream')):
        doc['status']='unsupported_content'
        doc['signals']=['non_text_body_not_interpreted']
        return doc
    text = body.decode('utf-8', errors='replace')
    links = []
    def link(target, title):
        target = urljoin(url, target)
        # DDG result links contain the actual public destination in uddg.
        parts = urlsplit(target)
        if parts.hostname and parts.hostname.endswith('duckduckgo.com'):
            target = dict(parse_qsl(parts.query)).get('uddg', target)
        try:
            target = clean_url(target)
        except ValueError:
            return
        if target not in {v['url'] for v in links} and len(links) < 35:
            links.append({'url': target, 'title': compact(title)[:180]})
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = ...
    if data is not ...:
        doc['content_kind'] = 'json'
        empty = data is None or data == [] or isinstance(data, dict) and ('data' in data and data['data'] in (None, [], {}))
        doc['status'] = 'empty_structured' if empty else 'readable'
        if empty: doc['signals'].append('explicit_empty_payload')
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    elif '<rss' in text[:2000].lower() or '<feed' in text[:1000].lower():
        doc['content_kind'], doc['status'] = 'discovery', 'readable'
        try:
            root = ET.fromstring(text)
            rows = []
            items=root.findall('.//item')
            atom=not items and root.tag.endswith('feed')
            if atom: items=root.findall('{*}entry')
            for item in items[:15]:
                if atom:
                    title=item.findtext('{*}title') or ''
                    target=next((v.get('href','') for v in item.findall('{*}link') if v.get('rel','alternate')=='alternate'),'')
                    description=BeautifulSoup(item.findtext('{*}summary') or '', 'html.parser').get_text(' ',strip=True)
                    if target: link(target,title)
                    rows.append(compact(title+' — '+description))
                    continue
                title = item.findtext('title') or ''
                target = item.findtext('link') or ''
                description = BeautifulSoup(item.findtext('description') or '', 'html.parser').get_text(' ', strip=True)
                link(target, title)
                rows.append(compact(title + ' — ' + description))
            text = chr(10).join(rows)
            doc['signals'].append('search_snippets_not_verified_facts')
        except ET.ParseError:
            text, doc['status'] = '', 'invalid_content'
    else:
        doc['content_kind'], doc['status'] = 'html', 'readable'
        soup = BeautifulSoup(text, 'html.parser')
        doc['title'] = compact(soup.title.get_text(' '))[:180] if soup.title else ''
        anchors = soup.find_all('a', href=True)
        anchors.sort(key=lambda a: 0 if a.find_parent(['main','article']) else 2 if a.find_parent(['nav','footer','header']) else 1)
        for a in anchors:
            link(a['href'], a.get_text(' '))
        structured = []
        scripts = soup.find_all('script')
        for s in scripts:
            if s.get('type') in {'application/ld+json', 'application/json'} or s.get('id') == '__NEXT_DATA__':
                try:
                    value = json.loads(s.string or s.get_text())
                    structured.append(json.dumps(value, ensure_ascii=False)[:12000])
                except (ValueError, TypeError):
                    continue
        for tag in soup(['script','style','noscript','nav','footer','header','form']): tag.decompose()
        # Preserve edit semantics used by rules sites instead of joining old/new numbers.
        for tag in soup.find_all(['del','deletedexplanation','s']):
            tag.replace_with('[DELETED TEXT: '+tag.get_text(' ',strip=True)+']')
            if 'revision_markers_present' not in doc['signals']: doc['signals'].append('revision_markers_present')
        for tag in soup.find_all(['ins','addedexplanation']):
            tag.replace_with('[INSERTED TEXT: '+tag.get_text(' ',strip=True)+']')
            if 'revision_markers_present' not in doc['signals']: doc['signals'].append('revision_markers_present')
        articles=soup.find_all('article')
        if len(articles)>1:
            text=chr(10).join(compact(a.get_text(' ',strip=True)) for a in articles)
            doc['signals'].append('multiple_article_sections')
        else:
            area = soup.find('article') or soup.find('main') or soup.body or soup
            text = compact(area.get_text(' ', strip=True))
        if len(text) < 700 and scripts and not structured:
            doc['signals'].append('sparse_static_text_possible_dynamic_page')
        if structured:
            text += chr(10) + 'EMBEDDED_PUBLIC_JSON:' + chr(10) + chr(10).join(structured)
            doc['signals'].append('embedded_structured_content')
        lowered = (doc['title'] + ' ' + text[:500]).lower()
        if any(term in lowered for term in ['verify you are human','just a moment','captcha','unusual traffic']):
            doc['status'] = 'access_blocked'
            doc['signals'].append('challenge_page_not_bypassed')
    if not text.strip() and doc['status'] == 'readable': doc['status'] = 'empty_text'
    doc['text_truncated'] = len(text) > 12000
    doc['text'] = text[:12000]
    doc['links'] = links
    return doc


class ExperienceWebTools:
    def __init__(self, archive, *, route='direct', session_factory=DiagnosticSession,
                 dns_check=check_public_dns):
        self.archive, self.route, self.session_factory, self.dns_check = archive, route, session_factory, dns_check

    def execute(self, action, *, run_id, reserve_request, cancelled=lambda: False):
        tool = action['tool']
        if tool == 'search_web':
            engine = action.get('engine') or 'bing'
            if engine not in SEARCH_ENGINES: raise ValueError('SEARCH_ENGINE_UNKNOWN')
            query = action.get('query', '').strip()
            if not query or len(query) > 360: raise ValueError('SEARCH_QUERY_INVALID')
            params = {'q': query}
            if engine == 'bing': params['format'] = 'rss'
            if engine == 'news': params.update({'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
            url = SEARCH_ENGINES[engine] + '?' + urlencode(params)
        elif tool == 'read_url':
            url = clean_url(action['url'])
        else:
            raise ValueError('UNKNOWN_TOOL')
        chain, seen = [], set()
        document = None
        for hop in range(3):
            if cancelled(): break
            url = clean_url(url)
            if url in seen: break
            seen.add(url)
            request_id = reserve_request(url)
            if request_id is None: break
            registered = registered_request(url)
            profile = request_header_profile(urlsplit(url)._replace(query='',fragment='').geturl())
            request_headers = dict(profile['headers'])
            if registered:
                request_headers.update(SPFTLL_HEADERS)
            started, status, headers, body, failure = time.monotonic(), None, {}, b'', None
            try:
                self.dns_check(url)
                with self.session_factory(self.route) as session:
                    r = session.get(url, headers=request_headers,
                                    timeout=(4, 12), allow_redirects=False)
                    status, body, headers = r.status_code, r.content, dict(r.headers)
            except (requests.RequestException, OSError, ValueError) as exc:
                failure = type(exc).__name__
            location = headers.get('Location') or headers.get('location')
            target = None
            if status in {301,302,303,307,308} and location:
                try: target = clean_url(urljoin(url, location))
                except ValueError: pass
            p = urlsplit(url)
            obs = self.archive.save(registered['source'] if registered else 'experience_web', urlunsplit((p.scheme,p.netloc,p.path,'','')),
                dict(parse_qsl(p.query,keep_blank_values=True)), body, status=status,
                headers=headers, request_id=request_id, error_class=failure,
                context={'run_id':run_id,'request_id':request_id,'url':url,
                         'redirect_target':target,'elapsed_seconds':time.monotonic()-started,
                         'source_adapter': 'spftll-agent-projection-v1' if registered else None})
            doc = project_body(body,url,status,headers.get('Content-Type',''),failure)
            doc = project_platform(obs, body, doc)
            doc.update({'document_id':'doc-'+obs.id,'raw_sha256':obs.content_hash,
                        'observed_at':obs.observed_at.isoformat(),'source_published_at':None,
                        'request_id':request_id})
            chain.append({'request_id':request_id,'url':url,'http_status':status,
                          'error':failure,'observation_id':obs.id,'redirect_target':target,
                          'elapsed_seconds':time.monotonic()-started})
            document = doc
            if not target: break
            url = target
        if document is None:
            return {'document':None,'requests':chain,'status':'cancelled_or_budget_exhausted'}
        document['redirect_chain'] = chain
        if tool == 'search_web':
            document['content_kind'] = 'discovery'
            if 'search_snippets_not_verified_facts' not in document['signals']:
                document['signals'].append('search_snippets_not_verified_facts')
        return {'document':document,'requests':chain,'status':'returned'}
