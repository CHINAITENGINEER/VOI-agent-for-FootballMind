from football_agent.return_observation import project_return_observation


def document(**values):
    base = {
        'document_id': 'doc-1', 'url': 'https://example.org/fixture', 'status': 'readable',
        'http_status': 200, 'content_kind': 'html', 'signals': [], 'text_truncated': False,
    }
    base.update(values)
    return base


def test_http_success_and_matching_content_do_not_claim_semantic_relevance():
    result = project_return_observation(document(), action={'action': 'read_url'}, acquisition_id='a1')
    assert result['transport_status'] == 'returned'
    assert result['evidence_surface'] == 'body_text'
    assert result['coverage'] == 'unknown'
    assert result['semantic_relevance'] == 'unassessed'
    assert result['information_gain'] is None and result['decision_answered'] is None
    assert result['truth_status'] == 'unverified'


def test_discovery_snippets_are_partial_navigation_surface():
    result = project_return_observation(document(content_kind='discovery', signals=['search_snippets_not_verified_facts']),
                                        action={'action': 'search_web'})
    assert result['evidence_surface'] == 'discovery_metadata'
    assert result['coverage'] == 'unknown'
    assert result['requires_agent_assessment'] is True
    assert result['semantic_relevance'] == 'unassessed'


def test_structured_truncation_is_exposed_as_partial_coverage():
    result = project_return_observation(document(content_kind='platform_structured',
        signals=['source_coverage_incomplete', 'bounded_source_order_projection_not_complete_result_set'],
        text_truncated=True))
    assert result['evidence_surface'] == 'structured_records'
    assert result['coverage'] == 'partial'
    assert result['signals'] == ['source_coverage_incomplete', 'bounded_source_order_projection_not_complete_result_set']


def test_failure_has_no_evidence_surface_and_does_not_need_semantic_assessment():
    result = project_return_observation(document(status='unavailable', http_status=403,
        content_kind='unknown', signals=['access_blocked'], error='HTTPError'))
    assert result['transport_status'] == 'failed_or_unavailable'
    assert result['evidence_surface'] == 'none'
    assert result['coverage'] == 'none'
    assert result['requires_agent_assessment'] is False
    assert result['semantic_relevance'] == 'unassessed'
