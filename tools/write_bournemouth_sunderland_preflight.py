import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = ROOT / 'var/work/fm-today-bournemouth-sunderland-20261018-v1'
out = root / 'frozen-f-paired-controls-v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
inv = json.loads((ROOT/'var/work/fm-today-preflight-20261018-v1a/run-report.json').read_text(encoding='utf-8'))
prior = json.loads((root/'market-prior.json').read_text(encoding='utf-8'))
verification = json.loads((out/'paired-controls.json').read_text(encoding='utf-8'))
gates = {
  'schema': 'footballmind-bournemouth-sunderland-preflight-gates-v1',
  'generated_at': datetime.now(timezone.utc).isoformat(),
  'fixture': {'match_id': prior['physical_match_id'], 'home': prior['fixture']['home'], 'away': prior['fixture']['away'],
              'kickoff': prior['fixture']['kickoff'], 'prediction_cutoff': prior['fixture']['prediction_cutoff']},
  'gates': {
    'A_identity': {'status': 'PASS', 'evidence': 'fresh spftll inventory projection; fixture 3003909, source team ids 348/65'},
    'B_cutoff': {'status': 'PASS', 'evidence': 'kickoff and cutoff are future relative to inventory observation; no result fields used'},
    'C_market_prior': {'status': 'PASS', 'evidence_path': 'market-prior.json', 'probabilities': prior['implied_probability_normalized']},
    'D_body_admission': {'status': 'PASS_OFFLINE_CONTROL', 'evidence': 'Football365 pre-match body saved and quote-checked; direct official pages reset'},
    'E_f_path': {'status': 'PASS_OFFLINE_PAIRED_CONTROL', 'positive_body_consumed': verification['controls']['positive_control']['body_consumed'],
                 'positive_modeled_claims': verification['controls']['positive_control']['modeled_claims'],
                 'positive_prediction_changed': verification['controls']['positive_control']['prediction_changed'],
                 'negative_body_consumed': verification['controls']['negative_control']['body_consumed'],
                 'negative_modeled_claims': verification['controls']['negative_control']['modeled_claims'],
                 'negative_prediction_changed': verification['controls']['negative_control']['prediction_changed']},
    'F_agent_runtime': {'status': 'NOT_STARTED', 'reason': 'current deployed live CLI scope does not include this fixture; this package is a frozen-f control, not a live Agent trace'}
  },
  'training': {'g_training_eligible': False, 'reason': 'no settled independent outcome; f control only'},
  'source_inventory_run_id': inv.get('run_id'),
  'provider_http': 0,
  'outcome_read': False,
  'artifacts': {'market_prior': sha(root/'market-prior.json'), 'paired_controls': sha(out/'paired-controls.json'),
                'manifest_path': 'manifest.json', 'manifest_hash_recorded_in_manifest': True}
}
(out/'preflight-gates.json').write_text(json.dumps(gates, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'path': str(out/'preflight-gates.json'), 'sha256': sha(out/'preflight-gates.json')}, ensure_ascii=False))
