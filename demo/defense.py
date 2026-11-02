"""Local HTTP facade for the separately frozen portable research release."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
from pathlib import Path
import threading
import uuid
from .defense_budget import MAX_READS, run_with_limit, validate_limit

ROOT=Path(__file__).resolve().parents[1]
RELEASE_ID='footballmind-defense-20261025-rc1'
DEFAULT_RELEASE=ROOT/'var/releases'/RELEASE_ID
POLICIES={'fulltext_mlp_seed20261027','state_ridge','fixed_order','all_evidence'}

class DefenseApplication:
    def __init__(self, release_dir=None, factory=None):
        self.release_dir=Path(release_dir or DEFAULT_RELEASE).resolve()
        self.factory=factory;self.backend=None;self.lock=threading.RLock();self.runs={}

    def _load(self):
        if self.backend is None:
            if self.factory:self.backend=self.factory(self.release_dir)
            else:
                path=self.release_dir/'footballmind_defense.py'
                if not path.is_file():raise FileNotFoundError('冻结研究包尚未准备完成')
                spec=importlib.util.spec_from_file_location('footballmind_frozen_defense_release',path)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                self.backend=module.FreezeApplication(bundle_dir=self.release_dir)
        return self.backend

    def catalog(self):
        with self.lock:
            status=dict(frozen_research_demo=True,release_id=RELEASE_ID,workspace_root=str(ROOT.resolve()),
                network_required=False,presentation_default='fulltext_mlp_seed20261027',development_selected_policy='state_ridge',
                max_reads=MAX_READS,default_max_reads=MAX_READS,original_frozen_budget=3,
                policies=[dict(id='fulltext_mlp_seed20261027',label='全文研究候选 · MLP seed27',basis='在三个全文 MLP 中按开发集收益 MSE 选择；不是整体最优策略'),
                    dict(id='state_ridge',label='开发集优选 · 状态 Ridge',basis='按开发集完整策略 Log Loss 选择；用于继续或停止，不区分当前候选正文'),
                    dict(id='fixed_order',label='固定顺序对照 · 非 g',basis='按冻结候选顺序，在同一预算内逐篇交给 f；不产生 g 评分'),
                    dict(id='all_evidence',label='按顺序尽量采用 · 非 g',basis='在本轮上限内按冻结顺序交给同一个 f；不产生 g 评分')],
                limitations=['历史重建的离线研究；不发起来源或语义 API 请求。','展示默认项与科学模型选择分别记录。','两种策略都尚未证明稳定优于同预算随机选择。'])
            try:status.update(release_ready=True,matches=self._load().catalog())
            except FileNotFoundError as exc:status.update(release_ready=False,matches=[],reason=str(exc))
            return status

    def case(self, case_id):
        with self.lock:return self._load().get_case(case_id)

    def evidence(self,case_id,document_id):
        with self.lock:return self._load().get_evidence(case_id,document_id)

    def start(self,payload):
        if set(payload)-{'case_id','model_name','max_reads'}:raise ValueError('仅接受比赛、冻结模型与正文上限选项')
        case_id=payload.get('case_id');model=payload.get('model_name','fulltext_mlp_seed20261027')
        limit=validate_limit(payload['max_reads']) if 'max_reads' in payload else None
        if not isinstance(case_id,str) or not case_id or len(case_id)>240:raise ValueError('case_id 无效')
        if model not in POLICIES:raise ValueError('只允许已冻结的研究模型')
        with self.lock:
            backend=self._load()
            result=backend.run_case(case_id,model_name=model) if limit is None else run_with_limit(backend,case_id,model,limit)
            run_id='df-'+uuid.uuid4().hex
            self.runs[run_id]=dict(run_id=run_id,case_id=case_id,model_name=model,state='COMPLETED',result=result,settlement=None)
            return deepcopy(self.runs[run_id])

    def get(self,run_id):
        with self.lock:
            if run_id not in self.runs:raise FileNotFoundError('本服务中没有该冻结演示记录')
            return deepcopy(self.runs[run_id])

    def settle(self,run_id,payload):
        if payload:raise ValueError('结算不能传入或修改赛果')
        with self.lock:
            record=self.get(run_id)
            if record['settlement'] is None:
                record['settlement']=self._load().settle_run(record['result'])
                self.runs[run_id]=record
            return deepcopy(record)
