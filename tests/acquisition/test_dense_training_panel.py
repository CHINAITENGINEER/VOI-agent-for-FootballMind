from tools.assemble_dense_training_corpus import enrich_panel


def c(a,h=None):return {'article_id':a,'body_sha256':h or a}


def test_fixed_groups_unchanged_and_leakage_removed():
    old=[{'match_id':'train','candidates':[c('a')]},{'match_id':'test','candidates':[c('t')]}]
    proposed=[{'match_id':'train','candidates':[c('a'),c('b'),c('leak','t')]}]
    panel,n=enrich_panel(old,proposed,{'train'})
    assert panel[1]==old[1]
    assert [x['article_id'] for x in panel[0]['candidates']]==['a','b']
    assert n==1


def test_old_retained_newest_added_cap_and_body_dedup():
    old=[{'match_id':'train','candidates':[c('old')]}]
    proposed=[{'match_id':'train','candidates':[c('old'),c('b'),c('c'),c('d'),c('alias','d')]}]
    panel,n=enrich_panel(old,proposed,{'train'},cap=3)
    assert [x['article_id'] for x in panel[0]['candidates']]==['old','c','alias']
    assert n==2
