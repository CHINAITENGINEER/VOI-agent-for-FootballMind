"""Real archived input/model checks for inference consumption, without HTTP."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from demo.live_f_consumption import consumption_audit, pipeline_summary
from football_research.acquisition import entity_evidence_updates as temporal
from football_research.acquisition import reviewed_rich_f as rich
from football_research.acquisition.semantic_f import public_input

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'var/parallel/agent/round2/runs/live-g/runs'
GERMAN='lg-764584968a2c435281bcf758efc8e9c8'
ITALIAN='lg-cfc68ef9dc6a4c3d9320574e7a452408'
NO_BODY='lg-5eac5b9b52314f828e178ccf17ea4077'
MODEL=ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').split('\n') if x.strip()]


def archive_case(run_id, step_index):
    run=read(RUNS/run_id/'run.json');step=run['steps'][step_index]
    before_ids=[s['input_id'] for s in run['steps'][:step_index]]
    after_ids=before_ids+[step['input_id']]
    def row(ids):return dict(physical_match_id=run['fixture']['physical_match_id'],prior=run['prior'],
        decision_at=step['executed_at'],used_input_ids=ids,
        evidence_events=[e for e in run['evidence_events'] if e['input_id'] in ids])
    model=temporal.load(RUNS/run_id/'f.json')
    before,after=row(before_ids),row(after_ids)
    np.testing.assert_allclose(temporal.predict(model,before),step['prediction_before'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(temporal.predict(model,after),step['prediction_after'],atol=1e-12,rtol=0)
    audit=consumption_audit(model,before,after,step['input_id'],step['body'],step['claims'],
                            step['prediction_before'],step['prediction_after'])
    return run,step,model,before,after,audit


def test_actual_german_body_has_modeled_features_and_nonzero_probability_response():
    run,step,model,before,after,audit=archive_case(GERMAN,0)
    assert step['body'] and audit['body_consumed']
    assert audit['claims_count']==3 and audit['modeled_claims']==2
    assert audit['new_nonzero_features']>0 and audit['model_feature_dimension']==1836
    assert audit['probability_l1_change']>1e-5
    assert audit['market_probability_l1_difference']>1e-5
    assert audit['reason']=='learned_evidence_residual'
    assert audit['body_sha256']==hashlib.sha256(step['body'].encode()).hexdigest()
    assert audit['training_performed'] is False
    assert audit['probability_changed_is_improvement'] is None


@pytest.mark.parametrize('index',[1,2])
def test_actual_german_empty_claim_body_is_consumed_without_fabricated_update(index):
    _,step,_,_,_,audit=archive_case(GERMAN,index)
    assert audit['body_consumed'] and step['f_executed']
    assert audit['claims_count']==0 and audit['modeled_claims']==0
    assert audit['probability_l1_change']<1e-12
    assert audit['reason']=='no_source_claims'
    assert audit['market_probability_l1_difference']>0 # Earlier valid evidence remains.


@pytest.mark.parametrize('index,reason',[(0,'no_source_claims'),(1,'no_supported_match_or_context_claims')])
def test_actual_inter_training_and_historical_text_explain_unchanged_prediction(index,reason):
    _,_,_,_,_,audit=archive_case(ITALIAN,index)
    assert audit['body_consumed'] and audit['reason']==reason
    assert audit['modeled_claims']==0 and audit['new_nonzero_features']==0
    assert audit['probability_l1_change']<1e-12


@pytest.mark.parametrize('claim_id',['S01-C01','S02-C02','S02-C03','S03-C05','S04-C01','S04-C03','S05-C02','S06-C03','S06-C04'])
def test_existing_reviewed_context_and_selection_activate_real_f_when_concept_is_preserved(claim_id):
    # Existing historical annotations are explicit sources for the concept.
    # This is a component diagnostic using original reconstructed Poisson priors.
    pool=ROOT/'var/work/fm-f-evidence-training-panel-20261014-01'
    samples={r['sample_id']:r for r in rows(pool/'samples.jsonl')}
    observed=next(r for r in rows(ROOT/'var/reviews/fm-f-evidence-panel-acceptance-20261014-v2/reviewed-observations.jsonl')
        if r['review_claim']['claim_id']==claim_id)
    review=observed['review_claim'];sample=samples[observed['sample_id']]
    body=(pool/sample['body_path']).read_text(encoding='utf-8')
    assert body[review['start']:review['end']]==review['quote']
    priors=read(ROOT/'var/reviews/fm-f-panel-reconstructed-priors-20261014-v1/structured-priors.json')
    prior=priors[sample['fixture']['physical_match_id']];fixture=prior['fixture']
    public=public_input(fixture,dict(url=sample['url'],title=sample['title'],text=body),'body')
    claim=dict(team=observed['team'],entity=observed['entities'][0],kind='other',value='unmodeled',
        scope='this_match' if review['scope'].startswith('this_match') else 'team_context',
        statement=review['assertion'],quotes=[review['quote']])
    model=temporal.load(MODEL)
    initial=dict(physical_match_id=fixture['physical_match_id'],decision_at=prior['as_of'],prior=prior['probabilities'],
        used_input_ids=[],evidence_events=[])
    before=temporal.predict(model,initial)
    def inspect(concept):
        event=dict(physical_match_id=fixture['physical_match_id'],input_id=public['input_id'],entity=claim['entity'],
            document_url=sample['url'],source_host=sample['publisher'],
            source_published_at=sample['original_clocks']['source_published_at'],historical_available_at=None,
            claim=claim,evidence=rich.evidence_tokens(claim,'body',fixture,concept=concept))
        event['event_id']=rich._digest(event)
        row={**initial,'used_input_ids':[public['input_id']],'evidence_events':[event]}
        after=temporal.predict(model,row)
        return consumption_audit(model,initial,row,public['input_id'],body,[claim],before,after)
    old=inspect(None);new=inspect(observed['observed_concept'])
    assert old['modeled_claims']==0 and old['nonzero_model_features']==0
    assert new['modeled_claims']==1 and new['new_nonzero_features']>0
    assert new['probability_l1_change']>1e-12


def test_screenshot_five_searches_are_not_five_bodies_or_updates():
    run=read(RUNS/NO_BODY/'run.json');summary=pipeline_summary(run)
    assert run['counts']['source']==5 and summary['body_documents']==0
    assert summary['semantic_requests']==0 and summary['f_body_updates']==0
    assert summary['changed_updates']==0 and summary['market_difference_l1']<1e-12
    assert summary['initialization_is_body_update'] is False


def test_pipeline_keeps_real_legacy_probability_changes_without_new_audit_fields():
    run=read(RUNS/GERMAN/'run.json');summary=pipeline_summary(run)
    assert summary['f_body_updates']==3
    assert summary['changed_updates']==1
    assert summary['modeled_claims'] is None
    assert summary['market_difference_l1']>1e-5


def test_same_http_success_does_not_establish_article_or_f_execution():
    run=dict(documents=[dict(status='OK',text='Directory',structured_news={'role':'index'}),
        dict(status='OK',text='',structured_news={'role':'article'})],
        steps=[],counts={'source':2,'body_attempts':0,'provider':0},current=[.5,.3,.2],prior=[.5,.3,.2])
    summary=pipeline_summary(run)
    assert summary['body_documents']==summary['f_body_updates']==0


def test_failed_step_is_not_counted_as_f_consumption():
    run=read(RUNS/NO_BODY/'run.json');run['steps']=[dict(f_executed=False,body='uncommitted')]
    assert pipeline_summary(run)['f_body_updates']==0
