"""Explicit single-attempt public web acquisition using verified Windows Schannel.

Only the registered Finnish host is supported. A route is chosen in trusted
configuration before I/O; failures never trigger a retry or another transport.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import time
from urllib.parse import urlsplit

from .audit import digest
from .inquiry_tools import OpenWebTools, public_url

VERSION = 'registered-public-web-windows-schannel-v1'
HOST = 'www.veikkausliiga.com'
_MATCH = re.compile(r'/tilastot/2026/veikkausliiga/ottelut/[1-9][0-9]*/')
_ARTICLE = re.compile(r'/uutiset/(?:(\d{4})/(\d{2})/(\d{2})/)?[a-z0-9][a-z0-9-]+/?')


def _validated(prepared, allowed_hosts):
    if (not isinstance(prepared,dict) or set(prepared)!={'tool','endpoint','params','source'}
            or prepared['tool']!='read_url' or prepared['source']!='open_page'
            or prepared['params']!={}):
        raise ValueError('System TLS accepts only a prepared public read without overrides or parameters')
    url=public_url(prepared['endpoint'],allowed_hosts)
    parts=urlsplit(url)
    if (parts.netloc!=HOST or url!=prepared['endpoint'] or parts.query or parts.fragment):
        raise ValueError('System TLS requires the registered exact HTTPS source')
    article=_ARTICLE.fullmatch(parts.path)
    if parts.path not in {'/','/uutiset/'} and _MATCH.fullmatch(parts.path) is None and article is None:
        raise ValueError('System TLS source path is not registered')
    if article is not None and article.group(1):
        date(*map(int,article.group(1,2,3)))
    return url,article is not None


def authorize_system_tls_prepared(prepared, public, allowed_hosts):
    """Retain the original task-seed/acquired-link capability boundary."""
    url,article=_validated(prepared,allowed_hosts)
    links={link['url'] for document in public['documents'] for link in document.get('links',[])}
    if url not in links and (article or url not in public['task']['seed_urls']):
        raise ValueError('System TLS article must be observed; other pages require a seed or acquired link')
    return url


class SystemTLSOpenWebTools(OpenWebTools):
    def __init__(self, archive, allowed_hosts, *, authorized_urls=(), clock=None, extraction_version=3):
        super().__init__(archive,allowed_hosts,clock=clock,extraction_version=extraction_version)
        self._authorized_urls=set(authorized_urls)
        for url in self._authorized_urls:
            _validated({'tool':'read_url','endpoint':url,'params':{},'source':'open_page'},self.allowed_hosts)

    @staticmethod
    def transport_policy():
        return {'version':VERSION,'route':'system_tls','registered_hosts':[HOST],
            'tls_verification':True,'default_system_certificate_revocation':True,
            'authentication':'none_no_netrc','proxy_policy':'explicit_direct_for_registered_source',
            'automatic_redirects':False,'request_attempts_per_action':1,'automatic_retries':0,
            'connect_timeout_seconds':6,'total_timeout_seconds':25,'max_response_bytes':2_000_000,
            'request_header_preset':'research-public-page-v1',
            'capability_policy':'task_seeds_or_observed_links_articles_require_observed_links',
            'candidate_may_override_policy':False}

    def configuration(self):
        return {**super().configuration(),'transport_policy':self.transport_policy()}

    def prepare(self, candidate, public):
        prepared=super().prepare(candidate,public)
        self._authorized_urls.add(authorize_system_tls_prepared(prepared,public,self.allowed_hosts))
        return prepared

    def execute(self, prepared, *, run_id, request_id):
        url,_=_validated(prepared,self.allowed_hosts)
        if url not in self._authorized_urls:
            raise ValueError('System TLS read has no capability from a verified prepare step')
        # Local version detection only; no identity/result capture is invoked.
        from football_research.data_adapters.system_tls_source import _client
        executable,client=_client()
        started=time.monotonic()
        requested_at=self.clock()
        status,error,exit_code,ssl_verify_result=None,None,None,None
        complete=True
        with TemporaryDirectory(prefix='football-public-web-') as temporary:
            body_path=Path(temporary)/'body.bin'
            headers_path=Path(temporary)/'headers.txt'
            command=[executable,'--disable','--noproxy','*','--proto','=https','--retry','0',
                '--connect-timeout','6','--max-time','25','--max-filesize','2000000',
                '--user-agent','FootballMind/0.1 research','--silent','--output',str(body_path),
                '--dump-header',str(headers_path),'--write-out','%{http_code}\t%{ssl_verify_result}',url]
            try:
                returned=subprocess.run(command,capture_output=True,timeout=30,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                exit_code=returned.returncode
                fields=returned.stdout.decode('ascii',errors='replace').strip().split('\t')
                status=int(fields[0]) if fields and fields[0].isdigit() and int(fields[0]) else None
                ssl_verify_result=fields[1] if len(fields)==2 else None
                error=None if exit_code==0 else 'CurlExit'+str(exit_code)
                if error is None and (ssl_verify_result!='0' or status is None or not 100<=status<=599):
                    error='SystemTLSVerificationOrStatusUnconfirmed'
            except (subprocess.TimeoutExpired,OSError) as exc:
                error=type(exc).__name__
            body=body_path.read_bytes() if body_path.exists() else b''
            complete=error is None
            if len(body)>2_000_000:
                body,complete,error=body[:2_000_000],False,'SystemTLSResponseLimit'
            headers={}
            if headers_path.exists():
                for line in headers_path.read_text(encoding='latin-1').splitlines():
                    key,separator,value=line.partition(':')
                    if separator and key.lower()=='content-type':
                        headers['content-type']=value.strip()
        transport={**deepcopy(self.transport_policy()),'client':client,
            'policy_sha256':digest(self.transport_policy()),'curl_exit_code':exit_code,
            'ssl_verify_result':ssl_verify_result,'primary_http_attempts':1,
            'system_certificate_metadata_requests':'not_measured'}
        observation=self.archive.save('open_page',url,{},body,status=status,headers=headers,
            error_class=error,observed_at=self.clock(),context={
                'inquiry_run_id':run_id,'inquiry_request_id':request_id,
                'requested_at':requested_at.isoformat(),'tool':'read_url',
                'response_complete':complete,'transport':transport})
        return {'document':self.project(observation,prepared),
            'observation':observation.model_dump(mode='json'),'request_id':request_id,
            'http_attempts':1,'latency_seconds':time.monotonic()-started,
            'money':None,'money_status':'not_measured'}


__all__=['VERSION','HOST','SystemTLSOpenWebTools','authorize_system_tls_prepared']
