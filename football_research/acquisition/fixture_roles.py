"""Deterministic entity mentions in already-visible text, not information value.

Unmentioned or ambiguous sides stay explicit. These roles describe mentions in
the original span; they do not assert causation, relevance or expected benefit.
"""
from copy import deepcopy
from hashlib import sha256
import re
import unicodedata

from .context_predictor import _state
from .state import parse_time
from football_spider.storage.raw_archive import digest

VERSION = 'verified-fixture-text-roles-v1'
ROLES = ('home','away','both','unattributed','ambiguous')
ROLE_NUMERIC = tuple('role_'+role+'_fraction' for role in ROLES)


def _normal(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def _mentions(text, aliases):
    normalized = _normal(text)
    names = {}
    for side in ('home','away'):
        for alias in aliases[side]:
            token = _normal(alias)
            if not token:
                raise ValueError('Verified role names must contain actual characters')
            names.setdefault(token,set()).add(side)
    hits=[]
    for name,sides in names.items():
        for match in re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)',normalized):
            hits.append({'start':match.start(),'end':match.end(),'name':name,'sides':sorted(sides)})
    # A canonical longer name wins over an entirely nested short name. Partial
    # overlap, or the same normalized name assigned to both sides, stays unknown.
    longest=[hit for hit in hits if not any(other['start']<=hit['start'] and hit['end']<=other['end']
                 and other['end']-other['start']>hit['end']-hit['start'] for other in hits)]
    longest.sort(key=lambda hit:(hit['start'],hit['end'],hit['name']))
    ambiguous=any(len(hit['sides'])!=1 for hit in longest) or any(
        one['start']<two['end'] and two['start']<one['end']
        for i,one in enumerate(longest) for two in longest[i+1:])
    sides={side for hit in longest for side in hit['sides']}
    role=('ambiguous' if ambiguous else 'both' if len(sides)==2 else
          next(iter(sides)) if sides else 'unattributed')
    return role,longest


def _identity_scope(value, identity):
    before=value['before']
    scope=identity['scope']
    actual={k:deepcopy(before['fixture'].get(k)) for k in ('home_team_id','away_team_id','competition_id','kickoff')}
    actual['kickoff']=parse_time(actual['kickoff']).isoformat()
    if (before['match_id']!=scope['match_id'] or actual!=scope['runtime_fixture']
            or parse_time(before['prediction_cutoff']).isoformat()!=scope['prediction_cutoff']
            or parse_time(scope['available_at'])>parse_time(before['decision_at'])):
        raise ValueError('Visible state and original ordered fixture identity scope disagree')
    return deepcopy(scope)


def _units(value):
    before=value['before']
    contents={row['content_id']:row['text'] for row in before['contents']}
    documents={row['document_index']:row for row in before['documents']}
    units=[]
    for doc in before['documents']:
        for index,ref in enumerate(doc['content_refs']):
            text=contents[ref['content_id']]
            if text.strip():
                units.append({'unit_id':f"document-{doc['document_index']}:content-{index}",
                    'kind':ref['role'],'text':text,'content_id':ref['content_id'],
                    'document_index':doc['document_index'],'document_id':doc['document_id'],
                    'raw_sha256':doc['raw_sha256'],'span':{k:ref[k] for k in ('start','end')}})
    links={}
    for link in before['observed_links']:
        if link['link_ref'] in links:
            raise ValueError('Visible observed links must have distinct parent references')
        links[link['link_ref']]=link
        doc=documents[link['document_index']]
        for index,anchor in enumerate(link['observed_anchors']):
            text=anchor['text']
            if text.strip():
                units.append({'unit_id':f"{link['link_ref']}:anchor-{index}",'kind':'link',
                    'text':text,'link_ref':link['link_ref'],'anchor_position':index,
                    'document_index':doc['document_index'],'document_id':doc['document_id'],
                    'raw_sha256':doc['raw_sha256']})
    for action_id,candidate in value['candidates'].items():
        refs=candidate.get('observed_link_refs',[])
        if (candidate.get('action_id')!=action_id or not isinstance(refs,list) or len(refs)!=len(set(refs))
                or any(ref not in links or links[ref]['target_url']!=candidate['request'].get('url') for ref in refs)):
            raise ValueError('Candidate role labels must reference its actual already-read parent links')
    return units


def _group(units, aliases):
    text_sets={role:set() for role in ROLES}
    projected=[]
    for unit in units:
        role,mentions=_mentions(unit['text'],aliases)
        text_sets[role].add(unit['text'])
        projected.append({**{k:deepcopy(v) for k,v in unit.items() if k!='text'},
            'text_sha256':sha256(unit['text'].encode()).hexdigest(),'role':role,'mentions':mentions})
    texts={role:sorted(text_sets[role]) for role in ROLES}
    counts={role:len(texts[role]) for role in ROLES}
    total=max(1,sum(counts.values()))
    return {'role_texts':texts,'role_counts':counts,'role_masks':{r:bool(counts[r]) for r in ROLES},
        'role_numeric':[counts[r]/total for r in ROLES],'units':projected}


def _projection(value, identity):
    scope=_identity_scope(value,identity)
    units=_units(value)
    state=_group(units,scope['aliases'])
    candidates={}
    for aid,candidate in value['candidates'].items():
        refs=set(candidate['observed_link_refs'])
        candidates[aid]=_group([u for u in units if u.get('link_ref') in refs],scope['aliases'])
    return {'schema':VERSION,'identity':identity,'identity_scope':scope,
        'visible_input_sha256':digest({k:value[k] for k in ('schema_version','before','candidates')}),
        'state':state,'candidates':candidates,
        'meaning':'observed_canonical_entity_mentions_not_relevance_or_predictive_value'}


def bind_fixture_roles(visible_state, identity):
    """Keep the existing encoded state and add only a recomputable sidecar."""
    from football_research.data_adapters.fixture_role_identity import verify_fixture_role_identity
    _state(visible_state,allow_missing_prior=True)
    checked=verify_fixture_role_identity(identity,as_of=visible_state['before']['decision_at'])
    result=deepcopy(visible_state)
    sidecar=_projection(result,checked)
    sidecar['projection_sha256']=digest(sidecar)
    if 'fixture_roles' in result and result['fixture_roles']!=sidecar:
        raise ValueError('An existing fixture role projection cannot be silently replaced')
    result['fixture_roles']=sidecar
    return result


def _checked(value):
    from football_research.data_adapters.fixture_role_identity import verify_fixture_role_identity
    state=_state(value,allow_missing_prior=True)
    sidecar=value.get('fixture_roles')
    if not isinstance(sidecar,dict) or sidecar.get('schema')!=VERSION:
        raise ValueError('An original-byte-verified fixture role sidecar is required')
    identity=verify_fixture_role_identity(sidecar['identity'],as_of=value['before']['decision_at'])
    expected=_projection(value,identity)
    expected['projection_sha256']=digest(expected)
    if expected!=sidecar:
        raise ValueError('Fixture roles differ from the actual visible text or original identity')
    portable={k:deepcopy(value[k]) for k in ('schema_version','before','candidates','fixture_roles')}
    if 'audit' in value:
        portable['audit']=deepcopy(value['audit'])
    return state,sidecar,portable


def role_state(visible_state):
    """Return the validated old model state plus five explicit mention roles."""
    state,sidecar,portable=_checked(visible_state)
    grouped=sidecar['state']
    return {**state,**{k:deepcopy(grouped[k]) for k in ('role_texts','role_counts','role_masks','role_numeric')},
        'identity_scope':deepcopy(sidecar['identity_scope']),
        'physical_match_id':sidecar['identity_scope']['physical_match_id'],
        'role_proof':{'schema':VERSION,'visible_state':portable}}


def candidate_role_state(visible_state, action_id):
    """Only pre-read link labels; neither request URL text nor hidden body."""
    _,sidecar,portable=_checked(visible_state)
    if action_id not in sidecar['candidates']:
        raise ValueError('An actual candidate action ID is required')
    grouped=sidecar['candidates'][action_id]
    return {'action_id':action_id,'texts':sorted({s for block in grouped['role_texts'].values() for s in block}),
        **{k:deepcopy(grouped[k]) for k in ('role_texts','role_counts','role_masks','role_numeric')},
        'identity_scope':deepcopy(sidecar['identity_scope']),
        'physical_match_id':sidecar['identity_scope']['physical_match_id'],
        'role_proof':{'schema':VERSION,'visible_state':portable,'action_id':action_id}}


def validate_role_state(state):
    proof=state.get('role_proof',{})
    if proof.get('schema')!=VERSION:
        raise ValueError('The original visible role proof is required')
    rebuilt=role_state(proof['visible_state'])
    if rebuilt!=state:
        raise ValueError('Role model input differs from its original visible proof')
    return rebuilt


def validate_candidate_role_state(state):
    proof=state.get('role_proof',{})
    if proof.get('schema')!=VERSION:
        raise ValueError('The original observed-parent role proof is required')
    rebuilt=candidate_role_state(proof['visible_state'],proof['action_id'])
    if rebuilt!=state:
        raise ValueError('Candidate role input differs from its observed-parent proof')
    return rebuilt


__all__=['VERSION','ROLES','ROLE_NUMERIC','bind_fixture_roles','role_state','candidate_role_state',
         'validate_role_state','validate_candidate_role_state']
