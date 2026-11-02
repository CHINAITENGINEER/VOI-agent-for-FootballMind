"""Interactive adoption limits over unchanged frozen models and candidate pools.

The original bundle remains byte-for-byte frozen. Explicit interactive requests
use a versioned protocol; the old three-document evaluation is still available
through the original bundle API and CLI.
"""
from copy import deepcopy
import time
import uuid

MAX_READS = 20
PROTOCOL = 'footballmind-interactive-postread-budget-v1'


def validate_limit(value):
    if type(value) is not int or not 1 <= value <= MAX_READS:
        raise ValueError('正文上限必须是 1–20 的整数')
    return value


def run_with_limit(backend, case_id, model_name, max_reads):
    """Score only the current state; cap adoption without forcing g to continue."""
    maximum = validate_limit(max_reads)
    with backend.lock:
        backend.verify()
        case = backend.cases[case_id]
        baseline = model_name in ('fixed_order', 'all_evidence')
        if model_name not in backend.config['models'] or (not baseline and model_name not in backend.g['fitted']):
            raise ValueError('Unknown frozen g head')
        setting = backend.config['models'][model_name]
        if maximum == setting['budget'] and model_name != 'all_evidence':
            result = backend.run_case(case_id, model_name=model_name)
            result.update(max_reads=maximum, requested_max_reads=maximum,
                          effective_adoption_limit=min(maximum,result['candidate_count']),
                          execution_protocol='original_frozen_budget', original_frozen_evaluation=True)
            backend.completed_runs[result['run_id']] = deepcopy(result)
            return result
        panel = backend._panel(case)
        mid = case['physical_match_id']
        evaluator = backend.FrozenEvaluator(backend.f, panel)
        ids = panel.sequences[mid]['document_ids']
        # There cannot be more remaining adoption opportunities than candidates.
        # Record this explicitly rather than presenting the UI cap as the pool size.
        budget = min(maximum, len(ids))
        threshold = setting['threshold']
        adopted = []; steps = []
        initial = evaluator.predict(mid, []).tolist()
        started = time.perf_counter()
        reason = 'budget_exhausted'
        for index in range(budget):
            remaining = [i for i in range(len(ids)) if i not in adopted]
            before = evaluator.predict(mid, adopted).tolist()
            scores = None
            if baseline:
                candidate = remaining[0]
            else:
                features = backend.np.stack([backend.candidate_features(evaluator, mid, adopted, i, budget,
                    backend.g['embedding_dimension'])[0] for i in remaining])
                scores = backend.predict_head(backend.g['fitted'][model_name], features,
                    backend.g['target_mean'], backend.g['target_scale'])
                if not backend.np.isfinite(scores).all():
                    raise ValueError('Nonfinite frozen g scores')
                best = int(backend.np.argmax(scores)); candidate = remaining[best]
            selected_before = [ids[i] for i in adopted]
            row = dict(step=index+1, state_before=dict(adopted_ids=selected_before,
                remaining_budget=budget-len(adopted), remaining_candidates=len(remaining)),
                p_before=before, p_after=None, f_executed=False, used_input_ids_before=selected_before,
                used_input_ids_after=None, adopted_ids=selected_before, selected_document_id=None,
                threshold=threshold, candidate_scores=[] if baseline else [dict(document_id=ids[i],score=float(v)) for i,v in zip(remaining,scores)],
                stop_reason=None, tie_break='first candidate in frozen source order',
                all_candidate_scores_equal=None if baseline else bool(backend.np.max(scores)==backend.np.min(scores)),
                g_state_depth_outside_training_range=not baseline and len(adopted)>2,
                selection_mode='explicit_baseline_no_g_selection' if baseline else 'frozen_g_selection')
            if not baseline and scores[best] <= threshold:
                reason = 'predicted_gain_below_frozen_threshold'; row['stop_reason'] = reason; steps.append(row); break
            next_adopted = [*adopted, candidate]
            after = evaluator.predict(mid, next_adopted).tolist()
            adopted = next_adopted
            row.update(selected_document_id=ids[candidate], p_after=after, f_executed=True,
                       used_input_ids_after=[ids[i] for i in adopted], adopted_ids=[ids[i] for i in adopted])
            steps.append(row)
        if len(adopted) == len(ids):reason = 'pool_exhausted'
        result = dict(run_id='defense-'+uuid.uuid4().hex, case_id=case_id, physical_match_id=mid,
            model_name=model_name, model_role=setting['role'], p_before=initial,
            p_after=evaluator.predict(mid,adopted).tolist(), steps=steps,
            selected_ids=[ids[i] for i in adopted], unselected_ids=[ids[i] for i in range(len(ids)) if i not in adopted],
            stop_reason=reason, budget=budget, max_reads=maximum, requested_max_reads=maximum,
            effective_adoption_limit=budget, threshold=threshold, body_adoptions=len(adopted), candidate_count=len(ids),
            final_state_recomputed=True, no_new_evidence_at_stop=reason=='predicted_gain_below_frozen_threshold',
            g_readable_ids=list(ids), f_used_ids=[ids[i] for i in adopted],
            model_bindings=dict(f_sha256=backend.f.f_model_sha256,g_archive_sha256=backend.config['g_model_sha256'],encoder=backend.config['encoder']),
            elapsed_seconds=time.perf_counter()-started,status='STOPPED',scope='historical_interactive_budget_research',
            outcomes_used_in_prediction=False,training_calls=0,network_requests=0,
            current_head_reads_candidate_body_embedding=False if baseline else backend.g['fitted'][model_name].get('input')=='full',
            g_selection_executed=not baseline,same_budget_comparison=True,
            execution_protocol=PROTOCOL, original_frozen_budget=setting['budget'],
            budget_feature_definition='min(requested_max_reads, candidate_count) - adopted_count',
            g_training_budget_values=[1,2,3,6], budget_outside_training_range=budget>6,
            g_training_adopted_count_range=[0,2],
            g_scored_state_outside_training_range=any(row['g_state_depth_outside_training_range'] for row in steps),
            original_frozen_evaluation=False,
            limitations=list(backend.config['limitations'])+['交互采用上限实验；权重和停止阈值不变，预算特征按本场可用采用次数计算，不改写原冻结评估。'])
        backend.completed_runs[result['run_id']] = deepcopy(result)
        return result
