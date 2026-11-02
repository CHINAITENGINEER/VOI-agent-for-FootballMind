from copy import deepcopy

import pytest

from tools.verify_postread_qwen_stability import aggregate_predictions


def fixture():
    features={"a0":{"physical_match_id":"match-a"},
              "b0":{"physical_match_id":"match-b"},
              "b1":{"physical_match_id":"match-b"}}
    labels={k:{"realized_gain":0.0} for k in features}
    assignment={"match-a":"test","match-b":"test"}
    records=[dict(sample_id=k,physical_match_id=r["physical_match_id"],split="test",
                  realized_gain=0.0,predicted_gain=2.0 if k=="a0" else 1.0)
             for k,r in features.items()]
    return records,features,labels,assignment


def test_independent_match_mean_is_not_pooled_record_mean():
    records,features,labels,assignment=fixture()
    values=aggregate_predictions(records,features,labels,assignment)
    assert values["row_mse"]==2.0
    assert values["match_macro_mse"]==2.5
    assert values["rows"]==3 and values["matches"]==2


def test_prediction_order_does_not_determine_alignment():
    records,features,labels,assignment=fixture()
    expected=aggregate_predictions(records,features,labels,assignment)
    assert aggregate_predictions(records[::-1],features,labels,assignment)==expected


@pytest.mark.parametrize("corruption,match",[
    ("match","Prediction match differs"),("split","Prediction split differs"),
    ("label","Saved target differs"),("duplicate","Duplicate prediction ID"),
])
def test_misaligned_or_duplicated_prediction_cannot_pass(corruption,match):
    records,features,labels,assignment=fixture()
    if corruption=="match":records[0]["physical_match_id"]="match-b"
    elif corruption=="split":records[0]["split"]="validation"
    elif corruption=="label":records[0]["realized_gain"]=0.1
    else:records.append(deepcopy(records[0]))
    with pytest.raises(AssertionError,match=match):
        aggregate_predictions(records,features,labels,assignment)
