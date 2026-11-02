"""Historical article caps must prevent a ninth request, including failed reads."""
import pytest

from football_research.data_adapters.phase3_historical_panel import Adapter, write_rows


def test_per_fixture_article_cap_is_checked_before_network(tmp_path):
    rows = [{'physical_match_id': 'target', 'tool': 'read_url', 'status': 'failed'} for _ in range(8)]
    rows += [{'physical_match_id': 'other', 'tool': 'read_url'},
             {'physical_match_id': 'target', 'tool': 'search_web'}]
    write_rows(tmp_path, 'attempts.jsonl', rows)
    adapter = Adapter.__new__(Adapter)
    adapter.root = tmp_path
    adapter.plan = {'read_budget_per_match': 8}

    def unexpected_fetch(*args, **kwargs):
        pytest.fail('Per-fixture cap must reject before HTTP')

    adapter.fetch = unexpected_fetch
    with pytest.raises(ValueError, match='per-fixture article request cap'):
        adapter.read({'physical_match_id': 'target'}, {'url': 'https://www.fulhamfc.com/news/test'}, {})
