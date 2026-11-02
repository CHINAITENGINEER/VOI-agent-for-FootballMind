"""Bounded verification of timezone code in exact official-page script links."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import re
from urllib.parse import urljoin,urlsplit,parse_qsl,urlunsplit
from football_spider.storage.raw_archive import RawArchive,atomic_write,json_bytes
from football_spider.platform_config import PlatformSettings
from football_platform.collectors.http_transport import RequestTransport


def main():
    archive=RawArchive(PlatformSettings().data_dir)
    page=next(o for o in archive.observations('understat_probe') if o.id=='b7212198-7610-44d4-9f52-4a51dae38c05')
    html=archive.read_body(page).decode('utf-8')
    scripts=re.findall(r'<script[^>]+src=["\x27]([^"\x27]+)',html)
    observations=[]
    def save(**facts):
        observations.append(archive.save('understat_js',facts['endpoint'],facts['params'],facts['body'],status=facts['http_status'],
            headers=facts['response_headers'],observed_at=facts['observed_at'],observation_id=facts['observation_id'],
            request_id=facts['request_id'],error_class=facts['error_class'],context={'research_only':True,'discovered_from':page.id}))
    report={'page_observation_id':page.id,'scripts':[]}
    with RequestTransport('understat_js',on_attempt=save,max_attempts=1,request_budget=2,interval=3) as http:
        for src in scripts:
            if not any(name in src for name in ('js/main.min.js?','js/calendar.min.js?')):continue
            url=urlsplit(urljoin('https://understat.com/',src))
            body=http.get(urlunsplit((url.scheme,url.netloc,url.path,'','')),dict(parse_qsl(url.query)))
            text=body.decode('utf-8')
            snippets=[]
            for term in ('timezoneOffset','getTimezoneOffset','getTimezoneDate','datetime'):
                for m in list(re.finditer(re.escape(term),text))[:6]:
                    snippets.append(text[max(0,m.start()-100):m.start()+240])
            entry={'url':urlunsplit(url),'observation_id':observations[-1].id,'sha256':observations[-1].content_hash,'snippets':snippets}
            report['scripts'].append(entry)
            print(json.dumps(entry),flush=True)
    out=Path(__file__).resolve().parents[2]/'var/parallel/collectors/round2/timezone-evidence.json'
    atomic_write(out,json_bytes(report))


if __name__=='__main__':main()
