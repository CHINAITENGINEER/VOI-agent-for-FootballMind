"""Read-only public endpoint probes. No guessed authenticated APIs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import requests
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes
from football_spider.platform_config import PlatformSettings


def main():
    archive = RawArchive(PlatformSettings().data_dir)
    report = []
    for source, url in [
        ('understat_probe', 'https://understat.com/league/EPL/2025'),
        ('clubelo_probe', 'http://api.clubelo.com/2026-09-19'),
        ('fbref_probe', 'https://fbref.com/en/comps/9/Premier-League-Stats'),
    ]:
        try:
            response = requests.get(url, timeout=(10, 20))
            obs = archive.save(source, url, {}, response.content, status=response.status_code,
                               headers=response.headers, context={'research_only': True})
            entry = {'source': source, 'status': response.status_code, 'bytes': len(response.content),
                     'markers': {x: x in response.text for x in ['datesData', 'xG', 'Elo', 'Date']}, 'observation_id': obs.id}
        except requests.RequestException as exc:
            obs = archive.save(source, url, {}, b'', error_class=type(exc).__name__)
            entry = {'source': source, 'error': type(exc).__name__, 'observation_id': obs.id}
        report.append(entry)
        print(entry, flush=True)
    atomic_write(archive.root / 'reports' / 'source-probes.json', json_bytes(report))


if __name__ == '__main__':
    main()
