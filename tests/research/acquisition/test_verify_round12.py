"""Independent weighting and decision-boundary regressions for erased priors."""
from copy import deepcopy

import pytest

from football_research.acquisition.verify_round12 import LEGAL, reference_cells, reference_choice


def samples(n=12):
    output=[]
    for i in range(n):
        for acquired,aid in LEGAL:
            gain = (.01*i if not acquired else .03*i+.2) if aid == "results_view" else -.1
            output.append({"match_id":str(i),"kickoff":"2025-01-05T12:00:00Z" if i<6 else "2025-01-12T12:00:00Z",
                "action":{"action_id":aid},"acquired_action_ids":list(acquired),"delta":gain})
    return output


def test_tool_average_collapses_two_states_before_match_support_and_week_se():
    tool=reference_cells(samples(),"action_only")["results_view",()]
    assert tool["support"] == 12
    assert tool["expected_gain"] == pytest.approx(.21)
    assert tool["standard_error"] == pytest.approx(.06)
    stage=reference_cells(samples(),"acquired_set")
    assert stage["results_view",()]["expected_gain"] == pytest.approx(.055)
    assert stage["results_view",("xg_view",)]["expected_gain"] == pytest.approx(.365)
    assert stage["results_view",()]["standard_error"] == pytest.approx(.03)


def test_twenty_two_transitions_are_only_eleven_matches_of_support():
    tool=reference_cells(samples(11),"action_only")["results_view",()]
    assert tool["support"] == 11 and not tool["supported"]
    assert tool["expected_gain"] is None
    assert tool["standard_error"] is not None  # Descriptive SE does not authorize a query.


@pytest.mark.parametrize("mutation",["missing","duplicate"])
def test_no_silent_partial_state_weighting_or_duplicate_support(mutation):
    rows=samples()
    rows=rows[1:] if mutation == "missing" else rows+[deepcopy(rows[0])]
    with pytest.raises(ValueError):
        reference_cells(rows,"action_only")


def test_se_is_not_invented_for_one_calendar_block():
    rows=samples()
    for r in rows:
        r["kickoff"]="2025-01-05T12:00:00Z"
    cell=reference_cells(rows,"action_only")["results_view",()]
    assert cell["supported"] and cell["standard_error"] is None


def test_cost_adjusted_tie_break_and_rounded_nonpositive_stop():
    actions=[{"action_id":"results_view","cost":{"money":.02,"requests":1}},
             {"action_id":"xg_view","cost":{"money":.03,"requests":1}}]
    cells={(aid,()):{"expected_gain":gain} for aid,gain in (("results_view",.02),("xg_view",.025))}
    assert reference_choice(cells,"action_only",actions,(),.05,2)=="results_view"
    assert reference_choice(cells,"action_only",actions,("results_view",),.03,1)=="xg_view"
    assert reference_choice(cells,"action_only",actions,(),.019,2) is None
    cells["results_view",()]["expected_gain"] = .01+1e-14
    cells["xg_view",()]["expected_gain"] = .015
    assert reference_choice(cells,"action_only",actions,(),.05,2) is None


def shared_fixture():
    values={((),"results_view"):.04,((),"xg_view"):.03,
        (("results_view",),"xg_view"):-.01,(("xg_view",),"results_view"):.02}
    cells={(2,"action_only"):{(aid,()):{"expected_gain":.02} for aid in ("results_view","xg_view")},
        (2,"acquired_set"):{(aid,acq):{"expected_gain":g} for (acq,aid),g in values.items()}}
    rows=[]
    for mid in ("a","b"):
        for (acq,aid),g in values.items():
            if acq == ("results_view",):
                g=.05 if mid=="a" else -.02
            rows.append({"fold":2,"partition":"test","match_id":mid,"kickoff":"2025-01-05T12:00:00Z",
                "acquired_action_ids":list(acq),"action_id":aid,"prediction_before":[.4,.3,.3],
                "estimates":{"same_version":{"expected_gain":g}},"actual_gain":999.})
    return rows,cells,{"results_view":.02,"xg_view":.03}


def test_shared_state_comparison_separates_mask_changes_from_value_variation():
    from football_research.acquisition.analyze_round12 import shared_state_contrasts
    rows,cells,costs=shared_fixture()
    result=shared_state_contrasts(rows,cells,costs)
    assert len(result["states"])==6
    stats=result["reassessment_summary"]["test"]
    assert stats["tool-prior-mean"]["net_value_changed"]==0
    assert stats["stage-prior-mean"]["positive_to_nonpositive"]==2
    assert stats["frozen-version-mean"]["positive_to_nonpositive"]==1
    variation=[r for r in result["within_fold_value_variation"] if r["acquired_action_ids"]==["results_view"]]
    assert {r["strategy"]:r["unique_rounded_gain_values"] for r in variation}=={
        "tool-prior-mean":1,"stage-prior-mean":1,"frozen-version-mean":2}


def test_shared_state_choices_do_not_consume_realized_gains():
    from football_research.acquisition.analyze_round12 import shared_state_contrasts
    rows,cells,costs=shared_fixture()
    expected=shared_state_contrasts(rows,cells,costs)
    for row in rows:
        row["actual_gain"]=object()  # An unreadable downstream outcome still has no effect.
    assert shared_state_contrasts(rows,cells,costs)==expected
