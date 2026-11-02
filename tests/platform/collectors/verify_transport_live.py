"""One explicit live check of A's new transport using an already proven URL.

The output is private research evidence under var/parallel/collectors, not a
production RawArchive. This script is not part of automated pytest or scheduling.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from football_platform.collectors.http_transport import RequestTransport, TransportError
from football_platform.collectors.envelopes import response_count


def main():
    discovery = json.loads((ROOT/'var/parallel/collectors/understat-research/data-manifest.json').read_bytes())
    if discovery['url'] != 'https://understat.com/getLeagueData/EPL/2025' or discovery['http_status'] != 200:
        raise ValueError('This verification requires the previously discovered successful URL')
    output = ROOT/'var/parallel/collectors/transport-live'
    output.mkdir(parents=True,exist_ok=True)
    records = []
    def record_attempt(**facts):
        body = facts.pop('body')
        name = facts['observation_id']
        payload_path = output/(name+'.body')
        payload_path.write_bytes(body)
        metadata = {**facts,'payload':str(payload_path.relative_to(ROOT)),
                    'sha256':hashlib.sha256(body).hexdigest(), 'payload_bytes':len(body),
                    'research_only':True,'production_contract_bound':False}
        metadata_path = output/(name+'.metadata.json')
        metadata_path.write_text(json.dumps(metadata,default=str,indent=2),encoding='utf-8')
        records.append({**metadata,'metadata':str(metadata_path.relative_to(ROOT))})
    report = {'status':'FAILED','source':'understat','purpose':'new_http_component_live_smoke_only',
              'started_at':datetime.now(timezone.utc).isoformat(), 'requests_expected_max':1}
    with RequestTransport('understat',on_attempt=record_attempt,request_budget=1,max_attempts=1) as client:
        try:
            body = client.get(discovery['url'],headers={
                'Referer':'https://understat.com/league/EPL/2025','X-Requested-With':'XMLHttpRequest'})
            count = response_count('understat','league',body)
            report.update(status='OK',response_match_count=count,
                          same_content_as_initial_discovery=hashlib.sha256(body).hexdigest()==discovery['sha256'])
        except TransportError as exc:
            report.update(error_class=exc.category)
        report['stats'] = client.stats
    report['attempts'] = records
    (output/'report.json').write_text(json.dumps(report,default=str,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'attempts'}))
    return 0 if report['status']=='OK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
