from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
from football_spider.tools.pg_test_cluster import local_cluster
from football_spider.storage.postgres import Repository


@pytest.fixture(scope='session')
def pg_url():
    try:
        with local_cluster() as (url, _):
            yield url
    except FileNotFoundError:
        pytest.skip('PostgreSQL integration requires PG_BIN pointing at existing binaries')


@pytest.fixture
def repo(pg_url):
    result = Repository(pg_url)
    # Isolated validation cluster only, never the user's DATABASE_URL.
    result.conn.execute('DROP SCHEMA IF EXISTS fm CASCADE')
    result.migrate()
    yield result
    result.close()
