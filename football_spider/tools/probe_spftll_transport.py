import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import asyncio
from football_spider.config import settings
from football_spider.spiders.base import AiohttpJsonClient


async def main():
    async with AiohttpJsonClient(settings) as client:
        for url, params in [(settings.detail_api_url, {'gameKind': 0, 'matchId': 3003894}),
                            (settings.more_odds_api_url, {'mKind': 0, 'matchId': 3003894})]:
            try:
                payload = await client.get_json(url, params)
                print(url, payload.get('code'), flush=True)
            except Exception as exc:
                print(type(exc).__name__, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
