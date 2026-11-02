"""Run the existing consumer with its installed scientific Python environment."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from football_agent.collector_refresh import PlatformRefreshAdapter
from football_agent.platform_tools import PlatformTools
from football_spider.storage.raw_archive import atomic_write,json_bytes

def main():
    root=Path(__file__).resolve().parents[2];data=root/'football_spider/platform_data'
    out=root/'var/parallel/collectors/round2'
    inputs=json.loads(Path(sys.argv[1]).read_bytes())
    service=PlatformRefreshAdapter(data/'raw',out/'agent-adapter-check')
    reader=PlatformTools(service=service,time_mode='live',archive_root=data/'raw')
    result=reader.execute('refresh_information',inputs)
    atomic_write(out/'agent-consumer-result.json',json_bytes(asdict(result)))

if __name__=='__main__':main()
