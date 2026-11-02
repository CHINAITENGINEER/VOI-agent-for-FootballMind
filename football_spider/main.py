"""Daily entry point: fetch today through the next seven days."""

from __future__ import annotations
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import sys
from datetime import date, timedelta

from loguru import logger

from football_spider.config import Settings, settings
from football_spider.spiders.match_detail import MatchDetailSpider
from football_spider.spiders.match_list import MatchListSpider
from football_spider.storage.database import Database


def configure_logging(config: Settings) -> None:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    logger.add(
        config.log_path,
        level="INFO",
        rotation="20 MB",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
    )


async def collect(
    config: Settings,
    start_date: date,
    future_days: int,
    refresh: bool = False,
    limit: int | None = None,
    list_only: bool = False,
) -> tuple[int, int, int]:
    end_date = start_date + timedelta(days=future_days)
    run_key = f"{start_date.isoformat()}:{end_date.isoformat()}"
    logger.info("采集窗口: {} 至 {}（含首尾）", start_date, end_date)

    matches = MatchListSpider(config).fetch(start_date, end_date)
    logger.info("列表接口返回并通过日期筛选: {} 场", len(matches))

    success = 0
    failed = 0
    with Database(config.database_path) as database:
        for match in matches:
            database.upsert_match(match)

        if list_only:
            return len(matches), 0, 0

        completed = set() if refresh else database.completed_ids(run_key)
        pending = [match for match in matches if match.id not in completed]
        if limit is not None:
            pending = pending[:limit]
        logger.info("详情待采集: {} 场；断点已完成: {} 场", len(pending), len(completed))

        async with MatchDetailSpider(config) as spider:
            async def fetch_identified(match):
                try:
                    return match, await spider.fetch(match), None
                except Exception as exc:  # keep processing the remaining games
                    return match, None, exc

            tasks = [
                asyncio.create_task(fetch_identified(match)) for match in pending
            ]
            for task in asyncio.as_completed(tasks):
                match, enriched, error = await task
                if error is None and enriched is not None:
                    database.upsert_match(enriched, full_detail=True)
                    database.mark_progress(run_key, enriched.id, "done")
                    success += 1
                    logger.info(
                        "详情完成 [{}/{}] {} {} vs {}",
                        success + failed,
                        len(pending),
                        enriched.id,
                        enriched.home_team,
                        enriched.away_team,
                    )
                else:
                    failed += 1
                    database.mark_progress(run_key, match.id, "failed", str(error))
                    logger.error("详情任务失败 {}: {}", match.id, error)

        logger.info("SQLite 当前共保存 {} 场比赛", database.count_matches())
    return len(matches), success, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xx.spftll.cn 足球数据采集器")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--future-days", type=int, default=settings.future_days)
    parser.add_argument("--refresh", action="store_true", help="忽略当天断点并刷新详情")
    parser.add_argument("--limit", type=int, help="仅处理前 N 场详情（调试用）")
    parser.add_argument("--list-only", action="store_true", help="只采集列表")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(settings)
    try:
        listed, success, failed = asyncio.run(
            collect(
                settings,
                start_date=args.date,
                future_days=args.future_days,
                refresh=args.refresh,
                limit=args.limit,
                list_only=args.list_only,
            )
        )
    except KeyboardInterrupt:
        logger.warning("任务被中断，已写入的比赛和断点会保留")
        return 130
    except Exception:
        logger.exception("采集任务失败")
        return 1

    logger.info("采集结束: 列表 {}，详情成功 {}，详情失败 {}", listed, success, failed)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
