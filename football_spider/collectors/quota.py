"""Local daily request/credit budget, reserved before external requests."""
import json
import time
from filelock import FileLock
from football_spider.models.contracts import now
from football_spider.storage.raw_archive import atomic_write, json_bytes


def reserve(root, source, cost, daily_limit):
    path = root / 'quotas' / f'{source}-{now():%Y-%m-%d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        state = json.loads(path.read_bytes()) if path.exists() else {'reserved': 0, 'date_basis': 'UTC'}
        if state['reserved'] + cost > daily_limit:
            return False
        state['reserved'] += cost
        state['updated_at'] = now().isoformat()
        atomic_write(path, json_bytes(state))
        return True


def request_slot(root, source, interval):
    """Persist pacing across short CLI invocations, not just one HTTP session."""
    path = root / 'quotas' / f'{source}-pacing.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        state = json.loads(path.read_bytes()) if path.exists() else {}
        current = time.time()
        scheduled = max(current, state.get('next_epoch', current))
        if scheduled - current > 60:
            return None
        atomic_write(path, json_bytes({'next_epoch': scheduled + interval}))
        return max(0, scheduled - current)
