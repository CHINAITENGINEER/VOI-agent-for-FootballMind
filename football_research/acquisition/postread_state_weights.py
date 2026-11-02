"""Outcome-blind match/depth balancing for exhaustive post-reading transitions."""
from collections import Counter, defaultdict
import numpy as np
from football_research.acquisition.corpus_rolling_g import match_weights

MODES = ('match_uniform_edges', 'match_uniform_depths', 'match_reachable_depths')


def state_weights(match_ids, depths, *, mode, adoption_cap=3):
    if mode not in MODES or len(match_ids) != len(depths) or not match_ids:
        raise ValueError('Invalid mode or aligned transition metadata')
    if any(isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 0 for d in depths):
        raise ValueError('Depths must be nonnegative integers')
    if isinstance(adoption_cap, bool) or not isinstance(adoption_cap, int) or adoption_cap < 1:
        raise ValueError('Invalid adoption cap')
    if mode == 'match_uniform_edges':
        return match_weights(match_ids)
    count = Counter((m,d) for m,d in zip(match_ids,depths)
                    if mode != 'match_reachable_depths' or d < adoption_cap)
    levels = defaultdict(set)
    for m,d in count:levels[m].add(d)
    if set(levels) != set(match_ids):
        raise ValueError('A match has no reachable transition')
    n = len(levels)
    return np.array([1/(n*len(levels[m])*count[m,d]) if (m,d) in count else 0.
                     for m,d in zip(match_ids,depths)])
