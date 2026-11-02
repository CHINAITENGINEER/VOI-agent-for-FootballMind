import json
import pytest
from football_research.data_adapters.fulham_nuxt import literal_nuxt, extract_archived_news


def test_nuxt_parameter_literals_do_not_execute_code():
    assert literal_nuxt('window.__NUXT__=function(a,b){return{data:a,flag:b,x:void 0}}([1,"ok"],!1);') == {
        "data": [1, "ok"], "flag": False, "x": None}
    for value in ['alert("x")', 'a.constructor', '{x:1,x:2}', '[1]+[2]']:
        with pytest.raises(ValueError):
            literal_nuxt('window.__NUXT__=function(a){return{data:'+value+'}}(1);')


def test_nuxt_body_bound_to_canonical_and_visible_title():
    url = "https://www.fulhamfc.com/news/2025/august/14/preview/"
    post = {"title": "Preview", "publishedDateTime": "2025-08-14T15:25:08Z",
            "savedTimestamp": "2025-08-14T15:26:48Z",
            "content": [{"rowData": {"widgetType": "TextBlockWidget", "widgetData": {
                "content": "<p>Player is available.</p><script>hidden()</script><p>Training update.</p>"}}}]}
    html = ('<html><head><link rel="canonical" href="'+url+'"><meta property="og:title" content="Preview"></head>'
            '<body><h1>Preview</h1><script>window.__NUXT__=function(a){return{data:[{status:200,data:a}]}}('
            +json.dumps(post).replace('</', '<\\/')+');</script></body></html>')
    parsed = extract_archived_news(html, url)
    assert parsed["body"] == "Player is available.\n\nTraining update."
    assert parsed["source_modified_at"] is None
    assert parsed["body_version_at_historical_kickoff"] == "unknown"
    with pytest.raises(ValueError, match="identity"):
        extract_archived_news(html, url.replace("preview", "different"))


def test_nuxt_post_title_fallback_and_heading_text_blocks():
    url = "https://www.fulhamfc.com/news/2025/december/19/preview/"
    post = {"title": None, "postTitle": "Preview", "publishedDateTime": "2025-12-19T12:00:00Z",
            "content": [{"rowData": {"widgetType": "TextBlockWidget", "widgetData": {
                "content": "<h5>Manager preview</h5>"}}}]}
    html = ('<html><head><link rel="canonical" href="'+url+'"><meta property="og:title" content="Preview"></head>'
            '<body><h1>Preview</h1><script>window.__NUXT__=function(a){return{data:[{status:200,data:a}]}}('
            +json.dumps(post).replace('</', '<\\/')+');</script></body></html>')
    parsed = extract_archived_news(html, url)
    assert parsed["body"] == "Manager preview"
