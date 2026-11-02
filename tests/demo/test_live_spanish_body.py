from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest

from demo.live_spanish_body import project_spanish_directory, project_spanish_body
from football_spider.models.contracts import Observation


def test_real_ceuta_cards_keep_actual_titles_and_unknown_clock():
    root = Path(__file__).resolve().parents[2] / 'var/work/live-spanish-source-repair-20261025-v1'
    if not root.exists():
        import pytest
        pytest.skip('Local real response not included in source checkout')
    response = json.loads((root/'response-1.json').read_text(encoding='utf-8'))
    observation = SimpleNamespace(**response['observation'])
    raw = gzip.decompress(next((root/'raw/1').rglob('*.body.gz')).read_bytes())
    result = project_spanish_directory(response['document'], raw, observation)
    assert result['live_spanish_directory_projection']['cards'] == 8
    coach = next(row for row in result['links'] if '/jose-juan-romero-' in row['url'])
    assert coach['title'].startswith('José Juan Romero:')
    assert coach['source_publication_local_time'] == '2026-10-23'
    assert coach['source_published_at'] is None
    assert coach['source_available_at'] is None
    assert not coach['target_article_fetched_for_context']
    assert len(result['links']) == len(response['document']['links'])


def test_cross_card_titles_and_unbound_new_links_are_not_added():
    url = 'https://adceutafc.com/noticias/'
    obs = SimpleNamespace(endpoint=url, params={}, http_status=200, error_class=None)
    html = f'<html><head><link rel="canonical" href="{url}"></head><body><h2>Outside title</h2><div class="noticia-card listing-card"><div class="listing-card-link"><a class="jet-listing-dynamic-link__link" href="https://adceutafc.com/preview/">Leer más</a></div></div></body></html>'
    document = {'url':url,'links':[{'url':'https://adceutafc.com/preview/','title':'Leer más'}]}
    result = project_spanish_directory(document, html.encode(), obs)
    assert result['links'] == document['links']
    assert result['live_spanish_directory_projection']['cards'] == 0


def cadiz_record():
    root = Path(__file__).resolve().parents[2] / 'var/work/live-spanish-source-repair-20261025-v1'
    if not (root/'response-4.json').exists():
        pytest.skip('Local real response not included in source checkout')
    response=json.loads((root/'response-4.json').read_text(encoding='utf-8'))
    raw=gzip.decompress(next((root/'raw/4').rglob('*.body.gz')).read_bytes())
    return response,raw


def test_real_cadiz_visible_full_body_matches_literal_blocks():
    response,raw=cadiz_record()
    result=project_spanish_body(response['document'],raw,Observation.model_validate(response['observation']))
    assert result['structured_news']['role']=='article'
    assert len(result['text'])==1155
    assert len(result['structured_news']['segments'])==5
    assert 'próximo domingo ante el Ceuta' in result['text']
    assert 'Copiar enlace' not in result['text']
    assert result['source_published_at']=='2026-10-22T19:28:00+00:00'
    assert result['source_authors']==[]
    assert result['source_modified_at'] is None
    assert result['body_sha256']==hashlib.sha256(result['text'].encode()).hexdigest()
    for row in result['structured_news']['segments']:
        assert result['text'][row['start']:row['end']]==row['text']


@pytest.mark.parametrize('mutation',['canonical','heading_hidden','body_hidden','body_changed','future_publication','missing_literal_record'])
def test_cadiz_unbound_hidden_inconsistent_or_future_rejected(mutation):
    from bs4 import BeautifulSoup
    response,raw=cadiz_record();soup=BeautifulSoup(raw,'html.parser')
    if mutation=='canonical':soup.select_one('link[rel="canonical"]')['href']='https://www.cadizcf.com/noticias/other'
    elif mutation=='heading_hidden':soup.select_one('.MkModuleHeaderNews__title')['hidden']=''
    elif mutation=='body_hidden':soup.select_one('.MkContentBlocks__richText')['hidden']=''
    elif mutation=='body_changed':soup.select_one('.MkContentBlocks__richText p').string='A substituted unrelated paragraph.'
    else:
        node=soup.select_one('script#__NEXT_DATA__');value=json.loads(node.get_text())
        if mutation=='future_publication':value['props']['pageProps']['data']['news']['publishedAt']='2099-01-01T00:00:00Z'
        else:del value['props']['pageProps']['data']['news']
        node.string=json.dumps(value,ensure_ascii=False)
    result=project_spanish_body(response['document'],str(soup).encode(),Observation.model_validate(response['observation']))
    assert result['structured_news']['role']!='article'
    assert result['live_spanish_body_projection']['status']=='rejected'
