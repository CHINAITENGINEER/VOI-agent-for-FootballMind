"""Bind newly proposed actions to tool capabilities, never to empirical gains."""
from copy import deepcopy
import math
import unicodedata

from .audit import digest
from .experience_sources import SEARCH_ENGINES, VERSION as TRANSPORT_VERSION, clean_url
from .structured_sources import PROJECTION_VERSION as STRUCTURED_VERSION

VERSION = 'runtime-discovered-routes-v1'
LOCAL_VERSION = 'runtime-discovered-routes-v2'
POLICY_VERSION = 'state-bound-runtime-policy-v5'
BOUND = 'capability_bound_unmeasured'


def configuration(options, unit):
    if type(options) is not dict or set(options) not in (
            {'search_web', 'read_url'}, {'search_web', 'read_url', 'query_acquired'}):
        raise ValueError('DISCOVERY_EXACT_COST_PROFILE_REQUIRED')
    engines, read = options['search_web'], options['read_url']
    if type(engines) is not dict or set(engines) - set(SEARCH_ENGINES):
        raise ValueError('DISCOVERY_UNSUPPORTED_ENGINE')
    local = options.get('query_acquired')
    amounts = list(engines.values()) + ([] if read is None else [read]) + ([] if local is None else [local])
    try:
        valid_costs = bool(amounts) and all(
            type(c) in (int, float) and math.isfinite(c) and c >= 0 for c in amounts)
    except OverflowError:
        valid_costs = False
    if not valid_costs:
        raise ValueError('DISCOVERY_FINITE_NONNEGATIVE_COSTS_REQUIRED')
    result = {'version':VERSION, 'search_web':{k:float(v) for k,v in sorted(engines.items())},
        'read_url':None if read is None else float(read), 'cost_unit':unit,
        'transport_version':TRANSPORT_VERSION, 'structured_version':STRUCTURED_VERSION,
        'semantics':'execution_capability_and_declared_cost_not_empirical_support',
        'duplicate_scope':'normalized_action_across_all_accepted_decisions',
        'negative_guard':'current_registered_nonpositive_normalized_action',
        'numeric_voi_training_eligible':False}
    if 'query_acquired' in options:
        from .runtime_local_acquisition import VERSION as BINDING_VERSION
        from .local_evidence import VERSION as QUERY_VERSION
        result.update(version=LOCAL_VERSION, query_acquired=None if local is None else float(local),
                      local_query_version=QUERY_VERSION, local_binding_version=BINDING_VERSION)
    return result


def descriptor(identity):
    saved = (identity or {}).get('configuration', {}).get('discovered_routes')
    if saved is None:
        return None
    fields = ('search_web','read_url','query_acquired') if saved.get('version') == LOCAL_VERSION else ('search_web','read_url')
    expected = configuration({k:saved.get(k) for k in fields},
        identity['configuration']['model_descriptor']['loss_unit'])
    from .runtime_source_support import policy_contract
    if saved != expected or (identity.get('version') != POLICY_VERSION and policy_contract(identity) is None):
        raise ValueError('DISCOVERY_CONFIGURATION_MISMATCH')
    return saved


def payload(response):
    action = response.get('action')
    if action == 'query_acquired':
        from .runtime_local_acquisition import payload as local_payload
        return local_payload(response)
    if action == 'search_web':
        query, engine = response.get('query'), response.get('engine')
        if not isinstance(query, str) or not query.strip() or len(query)>360 or engine not in SEARCH_ENGINES:
            raise ValueError('DISCOVERY_SEARCH_FIELDS')
        return {'action':action, 'engine':engine, 'query':query}
    if action == 'read_url':
        url = clean_url(response.get('url'))
        if url != response.get('url'):
            raise ValueError('DISCOVERY_CANONICAL_URL_REQUIRED')
        return {'action':action, 'url':url}
    raise ValueError('DISCOVERY_UNSUPPORTED_ACTION')


def action_key(response):
    if response.get('action') == 'query_acquired':
        from .runtime_local_acquisition import key_payload
        return digest({'version':LOCAL_VERSION, 'action':key_payload(response)})
    value = payload(response)
    if value['action']=='search_web':
        value['query']=' '.join(unicodedata.normalize('NFKC',value['query']).casefold().split())
    return digest({'version':VERSION, 'action':value})


def accepted_keys(responses, *, include_local=False):
    result = set()
    for response in responses:
        if response.get('action') in {'search_web','read_url'}:
            result.add(action_key(response))
        elif include_local and response.get('action') == 'query_acquired':
            try:
                result.add(action_key(response))
            except ValueError:
                # Fixture identity queries use a different existing contract.
                pass
    return sorted(result)


def registered_request_payload(route, match_input):
    match = route['match']
    if match['action'] == 'search_web':
        return {'action':'search_web', 'engine':match['engine'],
                'query':match['query_template'].replace('{match_input}', match_input)}
    return match


def _url_origin(public, url):
    if url in public.get('seed_urls',[]):
        return {'kind':'seed_url','url':url}
    if any(entry.get('url')==url for entry in public.get('source_catalog',{}).get('entries',[])):
        return {'kind':'source_catalog','url':url}
    for doc in public.get('documents',[]):
        for index, link in enumerate(doc.get('links',[])):
            if link.get('url')==url:
                return {'kind':'observed_link','document_id':doc['document_id'],
                        'link_index':index,'url':url}
    raise ValueError('DISCOVERY_URL_NOT_OBSERVED')


def bind(public, response, config):
    """No IO and no claim of semantic relevance; quote/source checks run later."""
    from .state_bound_runtime_policy import _source_route
    request = payload(response)
    selected = [q for q in response.get('analysis_plan',[])
                if q.get('id')==response.get('active_question_id') and q.get('status') in {'open','uncertain'}]
    if len(selected)!=1 or not response.get('unknown_to_resolve','').strip() or not response.get('useful_if','').strip():
        raise ValueError('DISCOVERY_OPEN_INFORMATION_QUESTION_REQUIRED')
    if request['action']=='search_web':
        if request['engine'] not in config['search_web']:
            raise ValueError('DISCOVERY_ENGINE_COST_UNDECLARED')
        cost, origin = config['search_web'][request['engine']], {'kind':'search_tool'}
    elif request['action']=='query_acquired':
        if config.get('version') != LOCAL_VERSION or config.get('query_acquired') is None:
            raise ValueError('DISCOVERY_LOCAL_COST_UNDECLARED')
        from .runtime_local_acquisition import bind as local_bind
        from .local_evidence import VERSION as QUERY_VERSION
        cost, origin = config['query_acquired'], local_bind(public, response)
        source, version = 'local:spftll', QUERY_VERSION
    else:
        if config['read_url'] is None:
            raise ValueError('DISCOVERY_READ_COST_UNDECLARED')
        cost, origin = config['read_url'], _url_origin(public,request['url'])
    if request['action'] != 'query_acquired':
        source, version = _source_route(response)
    key = action_key(response)
    return {'route_id':'discovered:'+key, 'action_id':'discovered:'+key,
        'source_id':source,'source_version':version,'cost':cost,'unit':config['cost_unit'],
        'request':request,'action_key':key,'origin':origin,
        'question':deepcopy(selected[0]),'unknown_to_resolve':response['unknown_to_resolve'],
        'useful_if':response['useful_if'], 'capability_version':config['version'],
        'semantic_relevance':'unassessed','empirical_support':False}


def bindings(public, batch, config, registered_routes):
    from .state_bound_runtime_policy import _matches
    bound, excluded = {}, {}
    for candidate in batch.candidates:
        response = candidate.response
        actions = {'search_web','read_url'}
        if config.get('version') == LOCAL_VERSION:
            actions.add('query_acquired')
        if response.get('action') not in actions:
            continue
        # Ambiguous, wrong-version and negative registered routes never become
        # a new dynamic capability merely because empirical selection failed.
        if any(_matches(route,response,public.get('match_input','')) for route in registered_routes):
            continue
        try:
            bound[candidate.candidate_id] = bind(public,response,config)
        except ValueError as exc:
            excluded[candidate.candidate_id] = str(exc)
    return bound, excluded


def registered_guards(routes, intervals, match_input):
    """A known negative request stays negative under spelling normalization.

    Intervals remain policy-reported; replay checks frozen route/cost and
    transport bindings, not the empirical history behind these estimates.
    """
    from .state_bound_runtime_policy import _source_route
    guards = []
    for route in routes:
        request = registered_request_payload(route, match_input)
        if (route['source_id'],route['source_version']) != _source_route(request):
            continue
        value = intervals.get(route['route_id'],{})
        if (value.get('support_status')=='supported_interval' and value.get('upper_bound') is not None
                and value['upper_bound']<=route['cost']):
            try:
                key = action_key(request)
            except ValueError:
                # A template can exceed the runtime query limit after expansion.
                # An unavailable request supplies no guard for other candidates.
                continue
            guards.append({'action_key':key,'registered_route_id':route['route_id'],
                           'declared_cost':route['cost'],'upper_bound':value['upper_bound']})
    return sorted(guards, key=lambda r:r['registered_route_id'])
