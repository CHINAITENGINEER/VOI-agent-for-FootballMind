"""Replay fixed f independently and diagnose selection signal without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.corpus_text_f import FrozenCorpusTextPredictor, _unpack
from football_research.acquisition.postread_g_use import score, select
from tools.train_e_drive_rolling_g import Data
from tools.train_e_drive_postread_g import SelectorInputs
from football_research.acquisition.convex_text_f import SCHEMA as CONVEX_SCHEMA
from scipy.special import softmax


class IndependentConvexReplay(FrozenCorpusTextPredictor):
    """Reconstruct inputs without the training Data.feature/predict path."""
    def predict(self, match_id, read_article_ids):
        ids = set(read_article_ids)
        menu = self.panel[match_id]['candidates']
        assert len(ids) == len(read_article_ids)
        assert ids <= {c['article_id'] for c in menu}
        base = self.base_model
        numeric = (self.numeric[self.positions[match_id]].astype(float)-base['mean'])/base['scale']
        prior = softmax(numeric@np.asarray(base['coef']).T+base['intercept'])
        selected = [c for c in menu if c['article_id'] in ids]
        match = self.matches[match_id]
        sides = [[c for c in selected if self.corpus.articles[c['article_id']]['source_team']==match[side]]
                 for side in ('home_team','away_team')]
        features = list(numeric)+[len(s) for s in sides]
        features += [np.mean([c['days_before_match_date'] for c in s]) if s else 0. for s in sides]
        features += [sum(self.corpus.articles[c['article_id']]['host']==h for c in selected)
                     for h in self.model['host_order']]
        for side in sides:
            vectors = [self.corpus.read_evidence(c['article_id'],already_read_ids=ids)['body_vector'] for c in side]
            features.extend(np.mean(vectors,axis=0) if vectors else np.zeros(384))
        m = self.model
        z = (np.asarray(features)[m['columns']]-m['mean'])/m['scale']
        after = softmax(np.log(prior)+np.r_[z,1.]@np.asarray(m['coef'])) if ids else prior
        return {'probabilities':after.tolist()}


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--experiment', type=Path, required=True); a = ap.parse_args(); out = a.experiment
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((out/k).read_bytes()).hexdigest() == v for k,v in manifest.items())
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    groups = [set(protocol[k]) for k in ['f_fit_ids','g_fit_ids','g_validation_ids','evaluation_ids']]
    assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
    panel = rows(out/'panel.jsonl'); byid = {p['match_id']:p for p in panel}
    bodies = [{c['body_sha256'] for m in ids for c in byid[m]['candidates']} for ids in groups]
    assert all(not bodies[i]&bodies[j] for i in range(4) for j in range(i))
    model = json.loads((out/'f-body.json').read_text(encoding='utf-8'))
    f = (IndependentConvexReplay if model['schema']==CONVEX_SCHEMA else FrozenCorpusTextPredictor)(a.root,out)
    assert not set(f.base_model['fit_match_ids']) & set(byid)
    checked = {}; error = 0.; edges = 0; after_adoption = []
    for name in ['training','validation','test']:
        for r in rows(out/(name+'-transitions.jsonl')):
            mid = r['match_id']; q = r['selector_state']; lab = r['label']
            ids = [c['article_id'] for c in byid[mid]['candidates']]
            adopted = q['f_adopted_ids']; aid = q['candidate_id']
            assert aid not in adopted and q['g_seen_ids'] == ids
            assert set(q) == {'g_seen_ids','f_adopted_ids','candidate_id','probabilities_before','candidate_body_legal_input'}
            expanded = [i for i in ids if i in set(adopted)|{aid}]
            for used, expected in [(adopted,q['probabilities_before']),(expanded,lab['probabilities_after'])]:
                key = (mid,tuple(used))
                if key not in checked:
                    checked[key] = f.predict(mid,used)['probabilities']
                delta = float(np.max(abs(np.array(checked[key])-expected))); error = max(error,delta)
                assert delta < 1e-12
            y = lab['outcome_index']
            assert abs(lab['signed_gain']-(np.log(lab['probabilities_after'][y])-np.log(q['probabilities_before'][y]))) < 1e-12
            if adopted:
                after_adoption.append(float(np.max(abs(np.array(lab['probabilities_after'])-q['probabilities_before']))))
            edges += 1
    data = Data(a.root,out,panel); inputs = SelectorInputs(data)
    models = {m:json.loads((out/f'g-{m}-selected.json').read_text(encoding='utf-8')) for m in protocol['modes']}
    replay_count = 0
    for run in rows(out/'policy-traces.jsonl'):
        model = models[run['mode']]; mid = run['match_id']
        for step in run['trace']:
            before = checked[(mid,tuple(step['f_adopted_ids']))]
            x = np.array([inputs.vector(run['mode'],mid,step['f_adopted_ids'],aid,before) for aid in step['candidate_ids']])
            chosen, values = select(model,step['candidate_ids'],x)
            assert chosen == step['selected']
            np.testing.assert_allclose(values,step['estimated_gains'],atol=1e-12,rtol=0)
            replay_count += 1
    if f.model['schema']==CONVEX_SCHEMA:
        weights = dict(f_coefficient_norm=float(np.linalg.norm(f.model['coef'])),f_schema=CONVEX_SCHEMA)
    else:
        w1,b1,w2,b2 = _unpack(np.array(f.model['parameters']),f.model['input_dimension'],f.model['hidden'])
        weights = dict(f_w1_norm=float(np.linalg.norm(w1)),f_w2_norm=float(np.linalg.norm(w2)),f_bias=b2.tolist())
    result = dict(status='PASS',manifest_files=len(manifest),transitions=edges,
        independently_replayed_f_states=len(checked),max_probability_error=error,g_decisions_replayed=replay_count,
        match_and_body_splits_disjoint=True,after_outcome_gain_not_in_g_feature_contract=True,
        f_regularization=f.model['alpha'],**weights,nonempty_to_nonempty_edges=len(after_adoption),
        maximum_nonempty_to_nonempty_probability_shift=max(after_adoption),
        nonempty_edges_above_1e_10=sum(d>1e-10 for d in after_adoption),network=0,refit=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
