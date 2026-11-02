"""Online acquisition with the frozen post-read Qwen f/g pair.

Acquisition fills a bounded pool before g scores full texts. No parameter
updates, outcome inputs, synthetic probabilities or legacy f/g fallbacks.
Existing legacy run files remain readable and are never rewritten by migration.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import math
import os
import re
from pathlib import Path
import threading
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from .live_g import (LiveGApplication, LiveBackend, TERMINAL, ROOT, now, utc,
                     digest, read, save, _directory_candidate_reason,
                     _review_pending_host_links)
from .live_candidate_time import candidate_time_scope
from .live_f_consumption import pipeline_summary
from .live_postread_model import LivePostreadModel, F_SHA, G_SHA, DEFAULT_HEAD

MODE = 'postread_fulltext_v1'
VERSION = 'footballmind-live-postread-v1'
SOURCE_BUDGET = 40
PREFETCH_BUDGET = 20
RUN_MINUTES = 30
PREFETCH_MINUTES = 5
ONLINE_TOKEN_BUDGET = 4096
CODE_FILES = ('demo/live_postread.py', 'demo/live_postread_model.py', 'demo/live_g.py',
    'demo/live_sources.py', 'demo/live_article_projection.py', 'demo/live_venezia_body.py', 'demo/live_dutch_body.py', 'demo/live_spanish_body.py',
    'demo/live_official_directory_cards.py', 'demo/live_candidate_time.py', 'demo/live_coverage.py')
LOADED_AT = now()
LOADED_SHA = {p: digest(ROOT / p) for p in CODE_FILES}
LIMITATIONS = [
    '在线全文研究：先获取真实正文并本地 Qwen 编码，再由冻结 g 决定是否交给配套 f。',
    '使用冻结研究权重；运行时只更新证据状态，不训练 f/g 或读取赛果。',
    '新全文 MLP 未证明稳定优于同预算随机选择；跨联赛和实时来源分布尚未校准。',
    '全文采集与编码已发生的开销不会被计为 g 节省的成本。',
    '最多尝试20个正文链接、来源请求40次；失败和重复也占尝试次数，g仍可提前停止。',
    '缺失claim保持未知；本版 f 直接使用全文向量及可观测状态，不要求DeepSeek抽取。',
    '预测变化不等于预测改善，赛果和收益只可在独立赛后结算。',
]


class LivePostreadApplication(LiveGApplication):
    """Reuse proven source/identity contracts, replace selection and prediction."""
    def __init__(self, store_dir, *, root=ROOT, backend=None, clock=now, model=None):
        self.root, self.store_dir = Path(root), Path(store_dir)
        self.backend = backend or LiveBackend(catalog_root=self.store_dir / 'espn')
        self.clock = clock
        self.lock = threading.RLock()
        self.workers, self.cancel = {}, {}
        self.catalog_worker = None
        self.auto_catalog = backend is None
        self.refresh_not_before = None
        self.model = model
        self.bindings = {'f': F_SHA, 'g': G_SHA, 'g_head': DEFAULT_HEAD,
                         'pipeline_mode': MODE, 'external_semantic_api_required': False}

    def catalog(self):
        result = super().catalog()
        from .live_sources import profile as source_profile
        from .live_coverage import availability, summarize
        for match in result.get('matches', []):
            match['can_start'], match['availability_reason'] = availability(match, self.clock(), minimum_lead_minutes=5)
            p = source_profile({'canonical': {'home_name': match.get('home') or match.get('home_team'),
                'away_name': match.get('away') or match.get('away_team'),
                'competition_name': match.get('competition_key') or match.get('source_competition_name')}})
            match['article_source_count'] = len(p['seed_urls'])
            match['article_source_status'] = 'registered_directories' if p['seed_urls'] else 'search_only'
            match['article_retrieval_verified'] = False
        result['coverage'] = summarize(result['matches'])
        ready = LivePostreadModel.readiness() if self.model is None else {'ready': True}
        result.update(schema_version=VERSION, pipeline_mode=MODE, limitations=LIMITATIONS,
            encoder_ready=ready['ready'], encoder_status='local_ready' if ready['ready'] else 'local_artifact_missing',
            encoder_reason=None if ready['ready'] else '本地冻结模型或 Qwen 文件不完整',
            provider_required=False, max_reads=20, default_max_reads=20,
            source_request_budget=SOURCE_BUDGET, prefetch_body_budget=PREFETCH_BUDGET,
            provider_request_budget=0, run_timeout_minutes=RUN_MINUTES,
            minimum_start_lead_minutes=5, prefetch_minutes=PREFETCH_MINUTES, local_encoding_token_budget=ONLINE_TOKEN_BUDGET,
            default_cost_loss_units=0.0, default_model_name=DEFAULT_HEAD,
            model_choices=[{'id': DEFAULT_HEAD, 'label': '冻结全文 g · MLP seed27'},
                           {'id': 'state_ridge', 'label': '冻结状态 Ridge 对照'},
                           {'id': 'fixed_order', 'label': '固定顺序对照 · 不用 g'}])
        result['runtime_bindings'] = {'loaded_at': LOADED_AT, 'process_id': os.getpid(),
            'code_sha256': dict(LOADED_SHA),
            'reload_required': any(digest(ROOT / p) != value for p, value in LOADED_SHA.items())}
        return result

    @staticmethod
    def _summary(run):
        if run.get('pipeline_mode') != MODE:
            return pipeline_summary(run)
        steps = [s for s in run.get('steps', []) if s.get('f_executed')]
        pool = run.get('candidate_pool', [])
        return {'schema': 'footballmind-live-postread-body-pipeline-v1',
            'body_documents': sum(bool(p.get('text')) for p in pool),
            'body_attempts': run['counts']['body_attempts'], 'semantic_requests': 0,
            'encoded_documents': run['counts'].get('encoded', 0),
            'g_scored_unique': run['counts'].get('g_scored_unique', 0),
            'f_body_updates': len(steps), 'audited_updates': len(steps),
            'changed_updates': sum(float(np.abs(np.asarray(s['prediction_after'])-s['prediction_before']).sum()) > 1e-12 for s in steps),
            'modeled_claims': None, 'known_modeled_claims': 0, 'semantic_rejected_bodies': 0,
            'input_mode': 'qwen_fulltext_embedding',
            'market_difference_l1': float(np.abs(np.asarray(run['current'])-run['prior']).sum())
                if run.get('current') is not None and run.get('prior') is not None else None,
            'initialization_is_body_update': False, 'training_performed': False}

    def _save(self, run):
        run['updated_at'] = self.clock()
        run['revision'] += 1
        run['body_pipeline'] = self._summary(run)
        with self.lock:
            save(self._folder(run['run_id']) / 'run.json', run)

    def get(self, rid):
        with self.lock:
            run = read(self._folder(rid) / 'run.json')
            if run['state'] not in TERMINAL and not (rid in self.workers and self.workers[rid].is_alive()):
                run.update(state='INTERRUPTED', reason='服务中断；保留已归档正文和已提交预测', stage='需要新建运行')
                self._save(run)
            run['body_pipeline'] = self._summary(run)
            return run

    def start(self, payload):
        if set(payload)-{'match_id', 'max_reads', 'cost_loss_units', 'model_name'}:
            raise ValueError('仅接受比赛、采用上限、增量处理成本和冻结策略')
        limit = payload.get('max_reads', 20)
        cost = payload.get('cost_loss_units', 0.0)
        head = payload.get('model_name', DEFAULT_HEAD)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('正文采用上限必须为1至20整数')
        if type(cost) not in (float, int) or not math.isfinite(cost) or not 0 <= cost <= .01:
            raise ValueError('增量成本必须为0至0.01 loss单位')
        if head not in {DEFAULT_HEAD, 'state_ridge', 'fixed_order'}:
            raise ValueError('未冻结的策略')
        with self.lock:
            cat = self.catalog()
            if cat['runtime_bindings']['reload_required']:
                raise ValueError('代码已更新，请重新启动服务后运行')
            if not cat['encoder_ready']:
                raise ValueError('本地Qwen或冻结f/g模型尚未就绪')
            if not cat['updated_at'] or utc(self.clock())-utc(cat['updated_at']) > timedelta(minutes=30):
                raise ValueError('请先刷新未来比赛目录')
            match = next((m for m in cat['matches'] if m['match_id'] == payload.get('match_id')), None)
            if not match or not match['can_start']:
                raise ValueError('比赛不在可执行赛前目录内')
            if any(t.is_alive() for t in self.workers.values()):
                raise ValueError('已有在线实验执行中，请等待结束')
            rid = 'lg-'+uuid4().hex
            self._folder(rid).mkdir(parents=True)
            run = dict(schema_version=VERSION, pipeline_mode=MODE, run_id=rid,
                match=deepcopy(match), created_at=self.clock(), updated_at=self.clock(), revision=0,
                state='PREPARING', reason=None, stage='核验本地冻结模型与真实市场先验',
                model_bindings=deepcopy(self.bindings), model_name=head, limitations=LIMITATIONS,
                runtime_bindings=cat['runtime_bindings'], selection_authority='experimental_postread_frozen_g',
                production_ready=False, max_reads=limit, cost_loss_units=cost,
                source_request_budget=SOURCE_BUDGET, prefetch_body_budget=PREFETCH_BUDGET,
                provider_request_budget=0, run_timeout_minutes=RUN_MINUTES,
                prefetch_minutes=PREFETCH_MINUTES, local_encoding_token_budget=ONLINE_TOKEN_BUDGET,
                current=None, before=None, prior=None, prediction_cutoff=None, forecast_at=None,
                attempts=[], documents=[], menus=[], candidate_pool=[], decisions=[], steps=[],
                semantic_attempts=[], body_rejections=[], evidence_events=[], used_input_ids=[],
                host_reviews=[], dynamic_allowed_hosts=[], discovery_diagnostics=[], pending=None,
                counts={'source': 0, 'provider': 0, 'body_attempts': 0, 'encoded': 0, 'g_scored_unique': 0},
                outcome=None, gain=None, g_training_eligible=False,
                input_mode='qwen_fulltext_embedding',
                acquisition_order_policy='source_round_robin_recent_observed_publication_first',
                acquisition_is_g_selection=False, costs={'source_requests': 0, 'encoding_seconds': 0.0,
                    'real_currency_amount': None, 'downstream_declared_loss_cost': cost})
            self.cancel[rid] = threading.Event()
            self._save(run)
            worker = threading.Thread(target=self._work, args=(run,), daemon=True)
            self.workers[rid] = worker
            worker.start()
            return deepcopy(run)

    def _check(self, run):
        if self.cancel[run['run_id']].is_set():
            raise InterruptedError('用户停止；不再发出新请求或提交新预测')
        if utc(self.clock()) >= utc(run['created_at'])+timedelta(minutes=RUN_MINUTES):
            raise TimeoutError('本轮处理时间预算已到')
        if run.get('prediction_cutoff') and utc(self.clock()) >= utc(run['prediction_cutoff']):
            raise TimeoutError('已到赛前截止；迟到正文仅归档，未提交f更新')

    def _request(self, run, tools, action, seeds, purpose, *, allow_route_retry=True):
        self._check(run)
        if run['counts']['source'] >= SOURCE_BUDGET:
            raise TimeoutError('来源请求预算已用完')
        prepared = tools.prepare(action, {'task': {'seed_urls': seeds}, 'documents': run['documents']})
        attempt = dict(request_id=uuid4().hex, purpose=purpose, action=action,
                       requested_at=self.clock(), status='PENDING', response=None)
        if action['tool'] == 'search_web':
            attempt['transport_route'] = getattr(tools, 'search_route', getattr(tools, 'default_route', 'environment'))
        run['attempts'].append(attempt)
        run['counts']['source'] += 1
        run['costs']['source_requests'] = run['counts']['source']
        run['pending'] = attempt['request_id']
        self._save(run)
        try:
            returned = tools.execute(prepared, run_id=run['run_id'], request_id=attempt['request_id'])
        except Exception as exc:
            attempt.update(status='FAILED', error_class=type(exc).__name__, handoff_at=self.clock())
            run['pending'] = None
            self._save(run)
            self._check(run)
            return {'status': 'FAILED', 'links': [], 'text': '', 'document_id': None, 'url': action.get('url')}
        attempt.update(status=returned['document']['status'], response=returned, handoff_at=self.clock())
        run['pending'] = None
        run['documents'].append(returned['document'])
        self._save(run)
        self._check(run)
        if (allow_route_retry and action['tool'] == 'search_web' and attempt['status'] == 'FAILED'
                and hasattr(tools, 'default_route') and attempt['transport_route'] != 'direct'
                and run['counts']['source'] < SOURCE_BUDGET):
            tools.search_route = 'direct'
            attempt['retry_reason'] = 'search_transport_failed_try_direct_once'
            return self._request(run, tools, action, seeds, 'search_transport_retry', allow_route_retry=False)
        return returned['document']

    def _admit_links(self, run, doc, entries, seen, candidates, *, search=False):
        from .live_sources import article_link_reason, search_team_binding
        for link in doc.get('links', []):
            url, title = link.get('url', ''), str(link.get('title') or '').strip()
            reason = article_link_reason(url, title, entries, seen, search=True) if search else _directory_candidate_reason(url, title, run['fixture'], entries, seen)
            if reason == 'outside_article_link_shape' and link.get('directory_metadata_adapter') == 'footballmind-live-ceuta-observed-news-cards-v1':
                reason = article_link_reason(url, title, entries, seen, search=True)
                if reason is None and (re.search(r'\bfemenino\b', title, re.I) or re.search(r'\bCeuta\s*[‘\'\"]?B[’\'\"]?\b', title, re.I)):
                    reason = 'directory_other_squad_label'
            # These are observed directory/squad markers, not article value
            # judgments. Keep the rejected link and its reason in the menu.
            if reason is None and '/categorieen/' in urlsplit(url).path.lower():
                reason = 'directory_category_navigation'
            fixture_names = ' '.join(str(run['fixture'].get(k) or '') for k in ('home_name', 'away_name'))
            if reason is None and re.search(r'\b(?:jong|jeugdopleiding)\b', title, re.I) and not re.search(r'\b(?:jong|jeugdopleiding)\b', fixture_names, re.I):
                reason = 'directory_other_squad_label'
            if search and reason is None and not search_team_binding(title, url, run['fixture'], entries):
                reason = 'search_team_binding_missing'
            scope = candidate_time_scope(link, self.clock())
            if reason is None and scope['status'] in {'stale', 'future'}:
                reason = 'outside_live_publication_window'
            row = {**link, 'action_id': 'article-'+sha256(url.encode()).hexdigest()[:24],
                'observed_at': self.clock(), 'parent_document_id': doc.get('document_id'),
                'admission_reason': reason or 'observed_article_link', 'publication_time_scope': scope}
            run['menus'].append(row)
            if reason is None:
                seen.add(url)
                candidates.append(row)

    @staticmethod
    def _prefetch_order(candidates):
        groups = {}
        for row in candidates:
            groups.setdefault(urlsplit(row['url']).hostname, []).append(row)
        for rows in groups.values():
            rows.sort(key=lambda r: r.get('publication_time_scope', {}).get('calendar_day') or '', reverse=True)
        result = []
        while any(groups.values()):
            for rows in groups.values():
                if rows:
                    result.append(rows.pop(0))
        return result

    def _work(self, run):
        from .live_sources import profile as source_profile, discovery_queries
        folder = self._folder(run['run_id'])
        try:
            self._check(run)
            if self.model is None or self.model.head != run['model_name']:
                self.model = LivePostreadModel(head=run['model_name'])
            model = self.model
            run['model_bindings'] = {**model.bindings, 'f': F_SHA, 'g': G_SHA}
            save(folder / 'model-bindings.json', run['model_bindings'])
            self._save(run)
            manifest = self.backend.initialize(run['match'], folder / 'setup')
            run['initialization'] = manifest
            run['prediction_cutoff'] = manifest['market_initialization']['prediction_cutoff']
            self._check(run)
            binding = self.backend.identity(manifest, folder / 'identity', run['match'])
            run['identity'] = binding
            c = binding['canonical']
            run['fixture'] = {'physical_match_id': c['source_event_id'],
                'home_name': c['home_name'], 'away_name': c['away_name'],
                'home_team': c['home_name'], 'away_team': c['away_name'],
                'home_aliases': binding['aliases']['home'], 'away_aliases': binding['aliases']['away'],
                'kickoff': c['kickoff'], 'prediction_cutoff': run['prediction_cutoff'],
                'competition_name': c.get('competition_name') or run['match'].get('competition_key'),
                'season': c.get('season')}
            run['prior'] = manifest['market_initialization']['probabilities']
            initial = model.initial_prediction(run['fixture'], run['prior'])
            self._check(run)
            run.update(current=initial['probabilities'], before=initial['probabilities'],
                       forecast_at=self.clock(), state='RUNNING', stage='发现真实文章目录，建立全文预取池')
            profile = source_profile(binding)
            run['source_profile'] = profile
            entries = profile['seed_urls'][:4]
            tools = self.backend.tools(folder / 'sources', entries, profile['allowed_hosts'])
            if hasattr(tools, 'configuration'):
                run['source_transport'] = tools.configuration()
            candidates, seen = [], set()
            for entry in entries:
                action = {'tool': 'read_url', 'url': entry}
                doc = self._request(run, tools, action, entries, 'directory_discovery')
                self._record_discovery_diagnostics(run, action, doc)
                self._admit_links(run, doc, entries, seen, candidates)
                self._save(run)
            queries = discovery_queries(c)
            run['discovery_queries'] = queries
            # Unlike the old query-before-reading path, two directory links do
            # not suppress source discovery. Fill up to the declared body cap.
            for query in queries[:5]:
                if len(candidates) >= PREFETCH_BUDGET:
                    break
                action = {'tool': 'search_web', 'query': query}
                doc = self._request(run, tools, action, entries, 'search_discovery')
                _review_pending_host_links(run, tools, doc)
                self._record_discovery_diagnostics(run, action, doc)
                self._admit_links(run, doc, entries, seen, candidates, search=True)
                self._save(run)
            run['discovery_summary'] = {'admitted': len(candidates), 'g_scored': False,
                'pool_body_limit': PREFETCH_BUDGET, 'acquisition_is_g_selection': False}
            if not candidates:
                run.update(state='STOPPED', reason='discovery_no_candidates', stage='来源未形成可准入文章链接')
                return
            docs, vectors, body_hashes = [], [], set()
            ordered_candidates = self._prefetch_order(candidates)
            prefetch_started = utc(self.clock())
            for position, candidate in enumerate(ordered_candidates[:PREFETCH_BUDGET]):
                if docs and utc(self.clock()) >= prefetch_started + timedelta(minutes=PREFETCH_MINUTES):
                    run['prefetch_stop_reason'] = 'prefetch_processing_time_budget_reached'
                    run['unattempted_candidates'] = ordered_candidates[position:]
                    break
                self._check(run)
                run['counts']['body_attempts'] += 1
                run['stage'] = '采集候选正文：'+candidate['title']
                doc = self._request(run, tools, {'tool': 'read_url', 'url': candidate['url']}, entries, 'candidate_body_prefetch')
                row = dict(action_id=candidate['action_id'], candidate_id=candidate['action_id'],
                    document_id=doc.get('document_id'), title=candidate['title'], url=candidate['url'],
                    source_url=doc.get('url') or candidate['url'], raw_sha256=doc.get('raw_sha256'),
                    status='body_failed', f_adopted=False, g_read=False, prediction_after=None,
                    receipt_handoff_at=self.clock())
                row.update(http_status=doc.get('http_status'), source_status=doc.get('status'),
                    source_error_class=doc.get('error_class'),
                    body_projection_role=(doc.get('structured_news') or {}).get('role'),
                    body_projection_exclusions=(doc.get('structured_news') or {}).get('excluded', []))
                run['candidate_pool'].append(row)
                if doc.get('status') != 'OK' or (doc.get('structured_news') or {}).get('role') != 'article' or not doc.get('text', '').strip():
                    row['reason'] = 'no_admitted_article_body'
                    run['body_rejections'].append({**row, 'f_executed': False, 'gain': None})
                    self._save(run)
                    continue
                text = doc['text']
                body_sha = sha256(text.encode('utf-8')).hexdigest()
                publication_conflict = 'conflict' in str(doc.get('publication_status') or '').lower()
                row.update(text=text, body=text, text_sha256=body_sha,
                    published_at=None if publication_conflict else doc.get('source_published_at') or candidate.get('source_published_at'),
                    publication_status=doc.get('publication_status'),
                    source_publication_date=doc.get('source_publication_date'),
                    source_publication_metadata=(doc.get('live_dutch_body_projection') or {}).get('source_metadata_published_at'),
                    directory_published_at=candidate.get('source_published_at'),
                    source_available_at=doc.get('observed_at'), first_public_available_at=None,
                    source_availability_basis='this_version_received_not_first_public_availability',
                    source_response_observed_at=doc.get('observed_at'), captured_at=None,
                    source_domain=urlsplit(row['source_url']).hostname)
                # Publication statements are distinct from actual retrieval.
                # Never admit a publication explicitly beyond this cutoff.
                if row['published_at']:
                    try:
                        if utc(row['published_at']) > utc(run['prediction_cutoff']):
                            row.update(status='body_failed', reason='publication_after_cutoff')
                            self._save(run)
                            continue
                    except (ValueError, TypeError):
                        row['publication_clock_unverified'] = True
                if body_sha in body_hashes:
                    row.update(status='duplicate_body', reason='same_exact_body_already_in_pool')
                    self._save(run)
                    continue
                record = {'document_id': row['document_id'], 'text': text, 'text_sha256': body_sha,
                    'title': row['title'], 'source_url': row['source_url'], 'source_domain': row['source_domain'],
                    'published_at': row['published_at'], 'captured_at': row['captured_at'],
                    'source_available_at': row['source_available_at'], 'historical_version_unknown': False,
                    'source_type': 'official_club' if row['source_domain'] in {urlsplit(e).hostname for e in entries} else 'unknown',
                    'existing_claims': None, 'raw_sha256': row['raw_sha256']}
                from .live_sources import DIRECTORIES, SOURCE_NAME_ALIASES
                owners = []
                for side in ('home', 'away'):
                    team = run['fixture'][side+'_team']
                    names = [team, *run['fixture'].get(side+'_aliases', [])]
                    registered = [DIRECTORIES.get(SOURCE_NAME_ALIASES.get(name, name)) for name in names]
                    if any(entry in entries and urlsplit(entry).hostname == row['source_domain'] for entry in registered if entry):
                        owners.append(team)
                record['source_owner_team'] = owners[0] if len(owners) == 1 else None
                row.update(status='encoding', encoding_started_at=self.clock())
                run['active_body'] = {'action_id': row['action_id'], 'document_id': row['document_id'], 'status': 'qwen_encoding'}
                run['stage'] = '正文已获取：本地Qwen编码，尚未交给f'
                save(folder / 'bodies' / (body_sha+'.json'), record)
                self._save(run)
                try:
                    encoded = model.encode_document(record, max_input_tokens=ONLINE_TOKEN_BUDGET)
                except Exception as exc:
                    budget_limited = getattr(exc, 'reason', None) == 'body_encoding_token_budget_exceeded'
                    row.update(status='encoding_budget_skipped' if budget_limited else 'encoding_failed',
                        error_class=type(exc).__name__, reason=getattr(exc, 'reason', 'embedding_failed'),
                        token_count=getattr(exc, 'token_count', None), encoding_completed_at=self.clock())
                    run['active_body'] = None
                    run['body_rejections'].append({'action_id': row['action_id'], 'document_id': row['document_id'],
                        'reason': 'body_encoding_token_budget_exceeded' if budget_limited else 'embedding_failed', 'error_class': type(exc).__name__, 'f_executed': False,
                        'prediction_after': None, 'gain': None})
                    self._save(run)
                    continue
                self._check(run)
                row.update(status='encoded', encoding_completed_at=self.clock(), encoding=encoded['encoding'],
                    vector_sha256=encoded['encoding'].get('vector_file_sha256'),
                    input_id=body_sha, model_document_index=len(docs))
                docs.append(encoded['document'])
                vectors.append(encoded['vector'])
                body_hashes.add(body_sha)
                run['counts']['encoded'] += 1
                run['costs']['encoding_seconds'] += encoded['encoding'].get('inference_seconds', 0.0) if not encoded['encoding'].get('cache_hit') else 0.0
                save(folder / 'encodings' / (body_sha+'.json'), encoded['encoding'])
                run['active_body'] = None
                self._save(run)
            if not docs:
                run.update(state='STOPPED', reason='no_encoded_candidate_bodies', stage='没有可供全文g使用的正文；未生成正文after')
                return
            adopted = []
            pool = [p for p in run['candidate_pool'] if p['status'] == 'encoded']
            effective_budget = min(run['max_reads'], len(docs))
            run['effective_adoption_limit'] = effective_budget
            run['stage'] = '全文g根据当前状态与候选正文评分'
            while True:
                self._check(run)
                evaluation = model.evaluate(run['fixture'], run['prior'], docs, vectors, adopted,
                    budget=effective_budget, cost_loss_units=run['cost_loss_units'])
                if not np.allclose(evaluation['prediction']['probabilities'], run['current'], atol=1e-6, rtol=0):
                    raise ValueError('Frozen f current prefix changed before selection')
                chosen = evaluation['selected_index']
                scored = []
                for score in evaluation['candidate_scores']:
                    i = score['index']
                    p = pool[i]
                    p['g_read'] = run['model_name'] == DEFAULT_HEAD
                    scored.append({**score, 'action_id': p['action_id'], 'title': p['title'], 'url': p['url']})
                run['counts']['g_scored_unique'] = len({x['action_id'] for d in run['decisions'] for x in d.get('candidates', [])} | {x['action_id'] for x in scored})
                decision = {'decision_at': self.clock(), 'action_id': pool[chosen]['action_id'] if chosen is not None else None,
                    'selected_document_id': pool[chosen]['document_id'] if chosen is not None else None,
                    'reason': evaluation['stop_reason'] if chosen is None else 'fixed_order_baseline' if run['model_name'] == 'fixed_order' else 'positive_postread_gain',
                    'threshold': model.threshold, 'candidates': scored,
                    'p_before': evaluation['prediction']['probabilities'], 'used_input_ids_before': list(run['used_input_ids']),
                    'candidate_fulltexts_read': [p['document_id'] for p in pool] if run['model_name'] == DEFAULT_HEAD else [],
                    'candidate_fulltexts_available': [p['document_id'] for p in pool],
                    'state_before': {'adopted_document_ids': [pool[i]['document_id'] for i in adopted],
                        'probabilities': evaluation['prediction']['probabilities'],
                        'f_hidden_state': evaluation['prediction'].get('hidden_state'),
                        'remaining_budget': effective_budget-len(adopted)},
                    'g_selection_executed': run['model_name'] != 'fixed_order', 'pipeline_mode': MODE}
                decision['budget_outside_training_support'] = evaluation.get('budget_outside_training_support')
                decision['state_depth_outside_training_support'] = len(adopted) > 2
                run['decisions'].append(decision)
                self._save(run)
                if chosen is None:
                    run.update(state='STOPPED', reason=evaluation['stop_reason'], stage='全文选择结束；保留实际f输出')
                    break
                p = pool[chosen]
                run['active_body'] = {'action_id': p['action_id'], 'document_id': p['document_id'], 'input_id': p['text_sha256'], 'status': 'f_running'}
                run['stage'] = '固定顺序采用正文：冻结新f执行预测' if run['model_name'] == 'fixed_order' else 'g已选择正文：冻结新f执行预测'
                self._save(run)
                before = list(run['current'])
                after = model.predict(run['fixture'], run['prior'], docs, vectors, [*adopted, chosen])
                self._check(run)
                if after.get('f_model_sha256') != F_SHA or after.get('f_executed') is not True:
                    raise ValueError('Frozen f execution/model binding is missing or different')
                expected_ids = [pool[i]['document_id'] for i in [*adopted, chosen]]
                if after.get('used_input_ids') != expected_ids:
                    raise ValueError('Frozen f consumed a different set of body IDs')
                after_probs = after['probabilities']
                adopted.append(chosen)
                run['current'] = after_probs
                run['forecast_at'] = self.clock()
                run['used_input_ids'] = [pool[i]['text_sha256'] for i in adopted]
                p.update(f_adopted=True, prediction_after=after_probs, f_committed_at=run['forecast_at'])
                run['steps'].append({'step': len(adopted), 'action_id': p['action_id'], 'document_id': p['document_id'],
                    'title': p['title'], 'url': p['url'], 'body': p['text'], 'raw_sha256': p['raw_sha256'],
                    'text_sha256': p['text_sha256'], 'input_id': p['text_sha256'], 'used_input_ids': list(run['used_input_ids']),
                    'f_used_document_ids': expected_ids, 'state_before': decision['state_before'],
                    'state_after': {'probabilities': after_probs, 'f_hidden_state': after.get('hidden_state'),
                        'adopted_document_ids': expected_ids},
                    'prediction_before': before, 'prediction_after': after_probs, 'f_executed': True,
                    'input_mode': 'qwen_fulltext_embedding', 'executed_at': run['forecast_at'],
                    'model_bindings': run['model_bindings'], 'encoding': p['encoding'], 'claims': [],
                    'claims_status': 'not_required_for_embedding_f', 'loss_reduction': None,
                    'f_consumption': {'schema': 'footballmind-postread-f-consumption-v1', 'body_consumed': True,
                        'input_id': p['text_sha256'], 'body_sha256': p['text_sha256'], 'body_characters': len(p['text']),
                        'input_mode': 'qwen_fulltext_embedding', 'modeled_claims': None,
                        'probability_l1_change': float(np.abs(np.asarray(after_probs)-before).sum()),
                        'training_performed': False}})
                run['active_body']['status'] = 'f_committed'
                self._save(run)
        except InterruptedError as exc:
            run.update(state='CANCELLED', reason=str(exc), stage='已停止')
        except TimeoutError as exc:
            run.update(state='STOPPED', reason=str(exc), stage='处理截止；迟到结果不提交')
        except Exception as exc:
            run.update(state='FAILED', reason=type(exc).__name__, failure_detail=str(exc)[:500],
                       stage='本轮失败；保留此前真实预测，失败不生成after')
        finally:
            run['pending'] = None
            self._save(run)
