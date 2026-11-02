"""Read-only, provenance-preserving inputs for offline baseline research."""
from .types import ResearchDataset, ResearchMatch, ResearchOddsSnapshot
from .spftll import load_spftll_archive
from .jsonl import load_jsonl, save_jsonl
from .features import load_feature_snapshot
from .football_data_co_uk import build_projection, iter_rows

__all__ = ['ResearchDataset', 'ResearchMatch', 'ResearchOddsSnapshot',
           'load_spftll_archive', 'load_jsonl', 'save_jsonl', 'load_feature_snapshot',
           'build_projection', 'iter_rows']
