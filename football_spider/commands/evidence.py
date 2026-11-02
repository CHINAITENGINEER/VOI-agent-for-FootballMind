"""Attach test and implementation evidence without hardcoded pass claims."""
import xml.etree.ElementTree as ET
from football_spider.platform_config import ROOT


def implementation_evidence(source):
    path = ROOT / 'docs' / 'evidence' / 'platform-tests.xml'
    if source=='understat':path=ROOT.parent/'var/parallel/collectors/round2/round2-tests.xml'
    result = {'implementation_status': 'IMPLEMENTED' if source in {'spftll', 'football_data', 'odds_api','understat'} else 'NOT_IMPLEMENTED',
              'contract_test_status': 'NOT_RUN', 'contract_test_report': str(path)}
    if path.exists():
        try:
            suites = list(ET.parse(path).getroot().iter('testsuite'))
            tests = sum(int(s.attrib.get('tests', 0)) for s in suites)
            failures = sum(int(s.attrib.get('failures', 0)) + int(s.attrib.get('errors', 0)) for s in suites)
            skipped = sum(int(s.attrib.get('skipped', 0)) for s in suites)
            result['contract_test_status'] = 'PASSED_SUITE' if tests and failures == skipped == 0 else 'PARTIAL_SUITE'
            result['suite_tests'] = tests
            result['test_report_timestamp'] = suites[0].attrib.get('timestamp') if suites else None
        except (ET.ParseError, ValueError, OSError):
            result['contract_test_status'] = 'UNREADABLE_REPORT'
    return result
