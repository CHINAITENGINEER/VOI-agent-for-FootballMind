"""Platform configuration, independent of the backward-compatible scraper."""
from pathlib import Path
import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)


class PlatformSettings(BaseModel):
    environment: str = Field(default_factory=lambda: os.getenv('PLATFORM_ENVIRONMENT', 'production'))
    database_url: str = Field(default_factory=lambda: os.getenv('DATABASE_URL', ''), repr=False)
    football_data_token: str = Field(default_factory=lambda: os.getenv('FOOTBALL_DATA_API_TOKEN', ''), repr=False)
    odds_api_key: str = Field(default_factory=lambda: os.getenv('THE_ODDS_API_KEY', ''), repr=False)
    api_football_key: str = Field(default_factory=lambda: os.getenv('API_FOOTBALL_KEY', ''), repr=False)
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv('PLATFORM_DATA_DIR', ROOT / 'platform_data')))
    timezone: str = Field(default_factory=lambda: os.getenv('PLATFORM_TIMEZONE', 'Asia/Shanghai'))
    sources: list[str] = Field(default_factory=lambda: os.getenv('PLATFORM_SOURCES', 'spftll,football_data,odds_api').split(','))
    competitions: list[str] = Field(default_factory=lambda: os.getenv('PLATFORM_COMPETITIONS', 'PL,BL1').split(','))
    odds_sports: list[str] = Field(default_factory=lambda: os.getenv('PLATFORM_ODDS_SPORTS', 'soccer_epl,soccer_germany_bundesliga').split(','))
    odds_regions: str = Field(default_factory=lambda: os.getenv('PLATFORM_ODDS_REGIONS', 'uk'))
    odds_markets: str = Field(default_factory=lambda: os.getenv('PLATFORM_ODDS_MARKETS', 'h2h,spreads,totals'))
    max_requests: int = Field(default_factory=lambda: int(os.getenv('PLATFORM_MAX_REQUESTS', '80')), ge=1)
    odds_credit_budget: int = Field(default_factory=lambda: int(os.getenv('PLATFORM_ODDS_CREDIT_BUDGET', '30')), ge=0)
    detail_limit: int = Field(default_factory=lambda: int(os.getenv('PLATFORM_DETAIL_LIMIT', '10')), ge=0)
    daily_time: str = Field(default_factory=lambda: os.getenv('PLATFORM_DAILY_TIME', '07:30'))
    refresh_minutes: list[int] = Field(default_factory=lambda: [int(x) for x in os.getenv('PLATFORM_REFRESH_MINUTES', '180,60').split(',')])
    job_grace_minutes: int = Field(default_factory=lambda: int(os.getenv('PLATFORM_JOB_GRACE_MINUTES', '120')))

    def public(self):
        data = self.model_dump(mode='json', exclude={'database_url', 'football_data_token', 'odds_api_key', 'api_football_key'})
        data['configured'] = {k: bool(getattr(self, k)) for k in ('database_url', 'football_data_token', 'odds_api_key', 'api_football_key')}
        ZoneInfo(self.timezone)
        return data
