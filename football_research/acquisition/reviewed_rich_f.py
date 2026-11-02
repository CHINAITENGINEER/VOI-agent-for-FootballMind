"""Research probability head over frozen reviewed assertions and exact quotes.

The reviewer supplies semantics; this is NOT LLM fine-tuning or an embedding
model. Fold-local TF-IDF preserves reviewed language beyond coarse predicates.
URLs, match IDs, outcomes and declared information values are not feature fields.
Recognized entity/team names are redacted; quotes can retain other names.
No-evidence and wholly OOV evidence retain the supplied prior.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from .context_predictor import _digest
from .historical_predictor_selection import _optimize

VERSION='reviewed-assertion-rich-residual-f-v1'
CHANNELS=('title','body')
TEAMS=('home','away','unknown')
MAX_TERMS=128


def clean_quote(claim, fixture):
    text=' '.join(claim['quotes'])
    names={claim['entity'],fixture['home_name'],fixture['away_name'],
           *fixture.get('home_aliases',[]),*fixture.get('away_aliases',[])}
    for name in sorted(names,key=len,reverse=True):
        if name.strip():text=re.sub(r'(?<!\w)'+re.escape(name)+r'(?!\w)',' person_or_team ',text,flags=re.I)
    return ' '.join(text.lower().split())


def evidence_tokens(claim, channel, fixture, concept=None):
    if channel not in CHANNELS or claim['team'] not in TEAMS:raise ValueError('Unknown evidence role')
    return {'channel':channel,'team':claim['team'],
        'atoms':[f"predicate:{claim['kind']}:{claim['value']}",f"scope:{claim['scope']}"]+([f'concept:{concept}'] if concept else []),
        'text':clean_quote(claim,fixture)}


def unique_evidence(items):
    return [v for _,v in sorted({_digest(i):deepcopy(i) for i in items}.items())]


def encoder_for(rows):
    matches=defaultdict(dict)
    for row in rows:
        for e in row['evidence']:matches[row['physical_match_id']][_digest(e)]=e
        matches[row['physical_match_id']]
    corpus=[' '.join(e['text'] for e in v.values()) for _,v in sorted(matches.items())]
    v=TfidfVectorizer(ngram_range=(1,2),max_features=MAX_TERMS,sublinear_tf=True)
    try:
        v.fit(corpus);terms=v.get_feature_names_out().tolist();idf=v.idf_.tolist()
    except ValueError as exc:
        if 'empty vocabulary' not in str(exc):raise
        terms,idf=[],[]
    atoms=sorted({a for r in rows for e in r['evidence'] for a in e['atoms']})
    return {'terms':terms,'idf':idf,'atoms':atoms,'fit_match_ids':sorted(matches)}


def features(encoder, row):
    terms=encoder['terms'];atoms=encoder['atoms'];width=len(terms)+len(atoms)
    vector=np.zeros((len(CHANNELS),len(TEAMS),width))
    analyzer=TfidfVectorizer(ngram_range=(1,2)).build_analyzer()
    grouped=defaultdict(list)
    for e in unique_evidence(row['evidence']):grouped[e['channel'],e['team']].append(e)
    for (channel,team),items in grouped.items():
        lexical=np.zeros(len(terms));present={a for e in items for a in e['atoms']}
        for e in items:
            counts=Counter(analyzer(e['text']))
            local=np.asarray([(1+np.log(counts[t]))*idf if counts[t] else 0 for t,idf in zip(terms,encoder['idf'])])
            if np.linalg.norm(local)>0:local/=np.linalg.norm(local)
            lexical+=local
        if np.linalg.norm(lexical)>0:lexical/=np.linalg.norm(lexical)
        vector[CHANNELS.index(channel),TEAMS.index(team)]=np.r_[lexical,[float(a in present) for a in atoms]]
    base=vector.ravel();p=np.asarray(row['prior'],float)
    if p.shape!=(3,) or not np.all(np.isfinite(p)) or np.any(p<=0) or abs(p.sum()-1)>1e-10:raise ValueError('Strict positive normalized prior required')
    return np.r_[base,(p[0]-p[2])*base]


def predict(model,row):
    x=features(model['encoder'],row)
    coefficients=np.asarray(model['coefficients'])
    if coefficients.shape!=(len(x),3) or not np.all(np.isfinite(coefficients)):raise ValueError('Coefficient shape/values differ')
    return softmax(np.log(row['prior'])+x@coefficients)


def fit(rows,regularization):
    if not rows:raise ValueError('Training rows required')
    labels={};counts=Counter()
    for r in rows:
        mid=r['physical_match_id'];y=r['outcome']
        if type(y)is not int or y not in (0,1,2) or (mid in labels and labels[mid]!=y):raise ValueError('Consistent per-match labels required')
        labels[mid]=y;counts[mid]+=1
    encoder=encoder_for(rows)
    x=np.asarray([features(encoder,r) for r in rows])
    design=(type('Encoder',(),{'encoder':encoder})(),x,np.asarray([r['outcome'] for r in rows]),
        np.log([r['prior'] for r in rows]),np.asarray([1/(len(counts)*counts[r['physical_match_id']]) for r in rows]))
    fitted,optimizer=_optimize(design,regularization)
    return {'schema':VERSION,'encoder':encoder,'coefficients':fitted.coefficients,'regularization':regularization,
        'optimizer':optimizer,'production_ready':False,'calibration_status':'not_established',
        'representation':'frozen_reviewed_claims_with_fold_local_quote_tfidf_and_semantic_atoms',
        'llm_weights_trained':False}


def save(model,path):
    payload={**deepcopy(model),'model_sha256':_digest(model)}
    with Path(path).open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)


def load(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8'));digest=raw.pop('model_sha256')
    if raw.get('schema')!=VERSION or _digest(raw)!=digest:raise ValueError('Frozen model binding differs')
    return raw
