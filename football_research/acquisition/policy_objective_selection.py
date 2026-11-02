"""Select a research g representation by complete out-of-fold policy returns.

This is model selection, not counterfactual identification or a new g learner.
Each match has equal weight; repeated budgets/costs are correlated scenarios.
"""
from collections import defaultdict
import math

MODES = ('baseline', 'structured', 'combined')
BUDGETS = (1, 2, 3)
COSTS = (0., .001, .01)
FIELDS = {'representation', 'physical_match_id', 'budget', 'cost', 'expected_net', 'fit_match_ids'}


def select(records, validation_ids):
    if not validation_ids or len(validation_ids) != len(set(validation_ids)):
        raise ValueError('Unique nonempty validation match IDs required')
    ids=set(validation_ids); cells={}; grouped=defaultdict(list)
    for r in records:
        if set(r) != FIELDS or r['representation'] not in MODES or r['physical_match_id'] not in ids:
            raise ValueError('Exact out-of-fold policy record required')
        fit=r['fit_match_ids']; mid=r['physical_match_id']
        if not isinstance(fit,list) or len(fit)!=len(set(fit)) or mid in fit:
            raise ValueError('Validation match must be excluded from parameter fitting')
        if type(r['budget']) is not int or r['budget'] not in BUDGETS or isinstance(r['cost'],bool) or r['cost'] not in COSTS:
            raise ValueError('Predeclared budget/cost grid required')
        value=r['expected_net']
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Finite out-of-fold return required')
        key=r['representation'],mid,r['budget'],r['cost']
        if key in cells: raise ValueError('Duplicate policy cell')
        cells[key]=value;grouped[r['representation'],mid].append(value)
    required={(mode,mid,b,c) for mode in MODES for mid in ids for b in BUDGETS for c in COSTS}
    if set(cells)!=required: raise ValueError('Complete identical validation coverage required')
    by_match={mode:{mid:math.fsum(grouped[mode,mid])/len(grouped[mode,mid]) for mid in sorted(ids)} for mode in MODES}
    means={mode:math.fsum(by_match[mode].values())/len(ids) for mode in MODES}
    chosen=max(MODES,key=lambda mode:(means[mode],-MODES.index(mode)))
    return {'schema':'historical-policy-objective-selection-v1','selected':chosen,'mean_net':means,
            'per_match_grid_mean_net':by_match,'validation_match_ids':sorted(ids),
            'criterion':'equal-match mean net loss reduction over fixed nine budget/cost scenarios',
            'independent_sample_count':len(ids),'scenarios_are_not_independent_samples':True,
            'production_ready':False,'online_selection_allowed':False}
