"""Turn admitted evidence into bounded information-acquisition proposals.

This generator supplies capabilities and source-grounded questions, not utility
estimates. The existing VoI policy and exploration allowance own selection.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import re
from urllib.parse import unquote, urlsplit

from .learned_voi_web import MatchDocumentCandidates, _url, _web_action
from .match_evidence import build_match_evidence, _landing_page_role_v2
from .match_source_profiles import _NATIONAL_ENTRIES_V2


LEGACY_VERSION = "evidence-document-candidates-v1"
VERSION = "evidence-document-candidates-v2"
LOCAL_PATH_VERSION = "evidence-document-candidates-v3"
OFFICIAL_SUBDOMAIN_VERSION = "evidence-document-candidates-v4"
CONTENT_ROLE_VERSION = "evidence-document-candidates-v5"
HTML_BODY_VERSION = "evidence-document-candidates-v6"
OBSERVED_CONTEXT_VERSION = "evidence-document-candidates-v7"
DATED_ARTICLE_VERSION = "evidence-document-candidates-v8"
FINNISH_LEAGUE_VERSION = "evidence-document-candidates-v9"
FINNISH_DATED_ARTICLE_VERSION = "evidence-document-candidates-v10"
OFFICIAL_IDENTITY_VERSION = "evidence-document-candidates-v11"
VISIBLE_ORDER_VERSION = "evidence-document-candidates-v12"
CARD_ORDER_VERSION = "evidence-document-candidates-v13"
CMS_CARD_ORDER_VERSION = "evidence-document-candidates-v14"
SUPPORTED_VERSIONS = (LEGACY_VERSION, VERSION, LOCAL_PATH_VERSION, OFFICIAL_SUBDOMAIN_VERSION,
                      CONTENT_ROLE_VERSION, HTML_BODY_VERSION, OBSERVED_CONTEXT_VERSION, DATED_ARTICLE_VERSION,
                      FINNISH_LEAGUE_VERSION, FINNISH_DATED_ARTICLE_VERSION, OFFICIAL_IDENTITY_VERSION,
                      VISIBLE_ORDER_VERSION, CARD_ORDER_VERSION, CMS_CARD_ORDER_VERSION)
_FOLLOWUP_VERSIONS = {5: 'match-evidence-extraction-v4', 6: 'match-evidence-extraction-v5',
                      7: 'match-evidence-extraction-v5', 8: 'match-evidence-extraction-v5',
                      9: 'match-evidence-extraction-v5', 10: 'match-evidence-extraction-v5',
                      11: 'match-evidence-extraction-v5', 12: 'match-evidence-extraction-v5',
                      13: 'match-evidence-extraction-v5', 14: 'match-evidence-extraction-v5'}
_VISIBLE_ORDER = {
    'version': 'visible-source-metadata-order-v1',
    'order': ['numeric', 'observed_official_article', 'unvisited_away_news_index',
              'unvisited_home_news_index', 'other_seed', 'other_observed_link', 'search'],
    'article_cues': ['opponent_and_full_fixture_date', 'injury_or_training',
                     'opponent', 'full_fixture_date'],
    'metadata_fields': ['title', 'snippet', 'observed_anchors.text'],
    'date_cue_formats': ['YYYY-MM-DD', 'D Month YYYY', 'Month D, YYYY'],
    'tie_break': 'original_document_then_link_exposure_order',
    'source_rank_basis': 'eligible_link_first_exposure_before_caps',
    'directory_guard': 'existing_landing_page_route_guard_never_promote_index_to_article',
    'cap_policy': 'preserve_source_order_caps_before_metadata_sorting',
    'semantics': 'predeclared_lexical_order_not_relevance_truth_value_or_propensity',
    'target_body_read': False,
}
_INJURY_CUE = re.compile(
    r'\b(?:injur(?:y|ies|ed)|fitness|doubt(?:ful)?|suspend(?:ed|sion)?|suspension|'
    r'training|team news)\b|伤停|伤病|伤愈|出场存疑|停赛|恢复训练', re.I)
_INDEX_PARTS = {"category", "listing", "tag", "tags", "page", "archive", "search", "men", "women", "all", "club"}
_NON_ARTICLE = re.compile(
    r"(?:^|[-_/\s])(?:ticket(?:s|ing)?|hospitality|shop|store|merchandise|cookie(?:s)?|privacy|terms|"
    r"membership(?:s)?|foundation|competition-winner|sweepstakes|stadium-tour(?:s)?)(?:$|[-_/\s])", re.I)


def _article_url(url, title):
    """Conservative URL shape filter, not a semantic or importance judgement."""
    parts = [part for part in unquote(urlsplit(url).path).casefold().split("/") if part]
    if "news" not in parts:
        return False
    tail = parts[parts.index("news") + 1:]
    if not tail or tail[0] in _INDEX_PARTS:
        return False
    return not _NON_ARTICLE.search("/".join(tail) + " " + str(title))


def _dated_article_url(url, title):
    """An observed permalink capability; its date is not a publication claim."""
    path = unquote(urlsplit(url).path).casefold()
    matched = re.fullmatch(r'/([0-9]{4})/([0-9]{2})/(?:([0-9]{2})/)?([^/]+)/?', path)
    if not matched:
        return False
    year, month, day, slug = matched.groups()
    try:
        date(int(year), int(month), int(day or 1))
    except ValueError:
        return False
    navigation = _INDEX_PARTS | {'feed', 'rss', 'index', 'home', 'about', 'contact', 'calendar',
                                 'schedule', 'fixtures', 'results', 'standings', 'sitemap'}
    return (slug not in navigation and any(c.isalpha() for c in slug)
            and re.fullmatch(r'[\w]+(?:[-_][\w]+)*', slug) is not None
            and not _NON_ARTICLE.search(slug + ' ' + str(title)))


def _finnish_dated_article_url(url, title):
    """Observed league permalink shape; dates are capabilities, not truth."""
    parts = urlsplit(url)
    matched = re.fullmatch(r'/uutiset/([0-9]{4})/([0-9]{2})/([0-9]{2})/([a-z0-9][a-z0-9-]+)/?', parts.path)
    if not matched or parts.query or parts.fragment:
        return False
    year, month, day, slug = matched.groups()
    try:
        date(int(year), int(month), int(day))
    except ValueError:
        return False
    return (any(c.isalpha() for c in slug) and slug not in _INDEX_PARTS
            and not _NON_ARTICLE.search(slug + ' ' + str(title)))


def _visible_parent(document, public):
    """Only a parent actually available by this decision grants v10 links."""
    from football_research.acquisition.state import parse_time
    try:
        observed = parse_time(document['observed_at'])
        available = parse_time(document['available_at'])
        decision = parse_time(public['decision_at'])
        cutoff = public['state'].get('prediction_cutoff')
        return observed <= available <= min(decision, parse_time(cutoff) if cutoff else decision)
    except (KeyError, TypeError, ValueError):
        return False


class EvidenceDocumentCandidates(MatchDocumentCandidates):
    """Preserve numeric actions and add observations that can be queried next.

    Club entry points may expose player-specific articles without spelling out
    the club name. Trust is limited to the registered same-host news routes;
    it never makes article claims true or assigns a numerical expected gain.
    """

    version = VERSION

    def __init__(self, *, match_terms, seed_urls, allowed_hosts, numeric_actions,
                 web_cost, match, profile, max_discovered_links=12,
                 max_followup_queries=12, max_candidates=64, discovery_version=2):
        if type(discovery_version) is not int or discovery_version not in set(range(1, 15)):
            raise ValueError("Discovery version must be from 1 to 14")
        self._discovery_version = discovery_version
        self.version = SUPPORTED_VERSIONS[discovery_version - 1]
        super().__init__(match_terms=match_terms, seed_urls=seed_urls, allowed_hosts=allowed_hosts,
                         numeric_actions=numeric_actions, web_cost=web_cost,
                         max_discovered_links=max_discovered_links)
        if type(max_followup_queries) is not int or not 0 <= max_followup_queries <= 32:
            raise ValueError("At most 32 grounded follow-up queries")
        if type(max_candidates) is not int or not 1 <= max_candidates <= 64:
            raise ValueError("Candidate cap must be 1..64")
        if len(self.numeric_actions) > max_candidates:
            raise ValueError("Candidate cap cannot drop numeric actions")
        if not isinstance(profile, Mapping):
            raise ValueError("An explicit source profile is required")
        self.match = deepcopy(dict(match))
        self.identity = build_match_evidence(self.match, [])['identity']
        for index, side in enumerate(("home", "away")):
            if self.match_terms[index].casefold() not in {alias.casefold() for alias in self.identity['aliases'][side]}:
                raise ValueError("Candidate match terms must match the evidence identity")
        self.profile = deepcopy(dict(profile))
        self.max_followups, self.max_candidates = max_followup_queries, max_candidates
        self._club_entry_hosts = {}
        self._local_article_routes = {}
        self._official_article_hosts = {}
        self._source_entry_by_host = {}
        self._finnish_dated_entries = set()
        for entry in self.profile.get('entries', []):
            if entry.get('kind') not in {'club_home', 'news_index'}:
                continue
            canonical = _url(entry['url'], self.allowed_hosts)
            if canonical not in self.seed_urls:
                continue
            side = entry.get('side')
            matches = [team for team in self.profile.get('teams', []) if team.get('side') == side]
            if len(matches) != 1 or matches[0].get('canonical_name') != entry.get('club'):
                raise ValueError("Official entry must belong to an identified fixture team")
            self._club_entry_hosts[canonical] = urlsplit(canonical).hostname
            if discovery_version >= 5:
                self._source_entry_by_host[urlsplit(canonical).hostname] = canonical
            registered = _NATIONAL_ENTRIES_V2.get(entry.get('club'))
            if (discovery_version >= 3 and registered and canonical == registered[0]
                    and self.profile.get('version') in {'verified-match-source-profiles-v2', 'verified-match-source-profiles-v3'}
                    and matches[0].get('registration_status') == 'raw_verified_fixture'):
                self._local_article_routes[canonical] = registered[1]
            if (discovery_version >= 9 and canonical in {
                    'https://www.veikkausliiga.com/', 'https://www.veikkausliiga.com/uutiset/'}
                    and ((self.profile.get('version') == 'verified-match-source-profiles-v4'
                          and self.profile.get('identity_schema') == 'finland-stage-fixture-identity-v1')
                         or (discovery_version >= 11
                             and self.profile.get('version') == 'verified-match-source-profiles-v5'
                             and self.profile.get('identity_schema') == 'official-fixture-evidence-identity-v1'))
                    and matches[0].get('registration_status') == 'raw_verified_fixture'
                    and entry.get('source_scope') == 'verified_competition_shared_entry'):
                self._local_article_routes[canonical] = r'/uutiset/[a-z0-9][a-z0-9-]+/?'
                if discovery_version >= 10:
                    self._finnish_dated_entries.add(canonical)
            if (discovery_version >= 4 and canonical == 'https://canadasoccer.com/'
                    and entry.get('club') == 'Canada'
                    and self.profile.get('version') == 'verified-match-source-profiles-v3'
                    and matches[0].get('registration_status') == 'raw_verified_fixture'
                    and 'news.canadasoccer.com' in self.allowed_hosts):
                self._official_article_hosts[canonical] = 'news.canadasoccer.com'
                if discovery_version >= 5:
                    self._source_entry_by_host['news.canadasoccer.com'] = canonical

    def configuration(self):
        configuration = {**super().configuration(), 'version': self.version,
                'match': deepcopy(self.match), 'profile': deepcopy(self.profile),
                'max_followup_queries': self.max_followups, 'max_candidates': self.max_candidates,
                'discovery_rule': 'admitted_grounded_followups_and_observed_official_same_host_news_links',
                'relevance_status': 'source_bound_proposals_not_semantic_truth_or_value_estimates',
                'candidate_cap_order': 'numeric_then_generic_search_then_seeds_then_grounded_queries_then_links',
                'candidate_generation_complete': False}
        if self._discovery_version >= 2:
            configuration['discovered_link_order'] = 'document_then_observed_link_order_not_value_ranked'
        if self._discovery_version >= 3:
            configuration['discovery_rule'] += '_and_registered_local_article_paths'
        if self._discovery_version >= 4:
            configuration['discovery_rule'] += '_and_observed_registered_official_news_subdomain'
        if self._discovery_version >= 5:
            configuration.update(
                followup_evidence_version=_FOLLOWUP_VERSIONS[self._discovery_version],
                official_link_scope='actual_OK_documents_on_registered_official_hosts_only',
                content_role_policy='unknown_or_index_text_cannot_propose_queries_links_remain_capabilities')
        if self._discovery_version >= 7:
            from .candidate_context import SPEC
            configuration['candidate_context_projection'] = deepcopy(SPEC)
        if self._discovery_version >= 8:
            configuration['discovery_rule'] += '_and_observed_official_same_host_dated_permalinks'
            configuration['dated_permalink_scope'] = 'valid_calendar_path_only_not_publication_relevance_or_value'
        if self._discovery_version >= 9:
            configuration['discovery_rule'] += '_and_observed_registered_finnish_competition_news_routes'
            configuration['finnish_route_scope'] = 'profile4_verified_stage_identity_actual_same_host_links_not_seed_articles'
        if self._discovery_version >= 10:
            configuration['discovery_rule'] += '_and_observed_finnish_calendar_dated_article_paths'
            configuration['finnish_dated_route_scope'] = 'registered_same_host_visible_OK_parent_valid_calendar_not_publication_relevance_or_value'
        if self._discovery_version >= 11:
            configuration['official_identity_scope'] = 'profile5_raw_official_fixture_no_market_prior_required'
        if self._discovery_version >= 12:
            configuration['visible_metadata_order'] = deepcopy(_VISIBLE_ORDER)
            configuration['visible_metadata_order']['injury_cue_pattern'] = _INJURY_CUE.pattern
            configuration['candidate_cap_order'] = 'legacy_source_order_caps_then_visible_metadata_order'
        if self._discovery_version >= 13:
            configuration['visible_metadata_order']['bound_card_title_policy'] = 'explicit_observed_card_title_allowed_alongside_empty_anchor_labels'
        if self._discovery_version >= 14:
            configuration['visible_metadata_order']['cms_card_title_policy'] = 'response_local_literal_title_with_pointer_not_rendered_anchor_or_value'
        return configuration

    def candidate_contexts(self, public, actions):
        """Optional public sidecar; old menus and snapshots remain unchanged."""
        if self._discovery_version < 7:
            return None
        from .candidate_context import build_candidate_contexts
        context = build_candidate_contexts(public, actions, allowed_hosts=self.allowed_hosts)
        if self._discovery_version >= 12:
            _, ordering = self._visible_order(public, actions)
            context['ordering'] = ordering
        return context

    def actions_with_prior_features(self, public, features_by_action, *, field='synthetic_prior_features'):
        """Return this generator's menu with explicit pre-query g-prior features.

        Discovery remains lexical and source-bound.  The supplied mapping is a
        separate declaration from the controller; this method never derives a
        value from titles, URLs, or documents and never changes the serialized
        generator configuration.
        """
        from .g_prior_features import attach_any_prior_features
        return attach_any_prior_features(self(public), features_by_action, field=field)

    def actions_with_prequery_features(self, public, capabilities_by_action, *,
                                       schema='footballmind-g-prior-features-v1',
                                       field='synthetic_prior_features'):
        """Build prior vectors from public state and an explicit capability registry.

        The generator remains responsible only for source-grounded candidate
        discovery.  Feature semantics are declared by the caller before the
        query and state-dependent quantities are derived by the versioned
        pre-query builder; titles, URLs and returned documents are never read
        for value inference.
        """
        from .g_prior_features import attach_prequery_prior_features
        return attach_prequery_prior_features(
            self(public), public, capabilities_by_action, schema=schema, field=field)

    def actions_with_strict_prequery_features(self, public, capabilities_by_action, *,
                                              schema='footballmind-g-teaching-features-v1',
                                              field='synthetic_prior_features'):
        """Build a menu using the canonical v2 pre-query public-state contract.

        This opt-in entrypoint is for new runs.  Historical replay keeps the
        v1 ``actions_with_prequery_features`` path so old snapshots remain
        reproducible; neither path reads titles, URLs, or returned documents
        to manufacture a prior feature.
        """
        from .g_prior_features import attach_strict_prequery_prior_features
        return attach_strict_prequery_prior_features(
            self(public), public, capabilities_by_action, schema=schema, field=field)

    @classmethod
    def from_configuration(cls, configuration):
        if not isinstance(configuration, Mapping) or configuration.get('version') not in SUPPORTED_VERSIONS:
            raise ValueError("Unsupported evidence candidate configuration")
        fields = ('match_terms', 'seed_urls', 'allowed_hosts', 'numeric_actions', 'web_cost',
                  'match', 'profile', 'max_discovered_links', 'max_followup_queries', 'max_candidates')
        try:
            generator = cls(**{key: deepcopy(configuration[key]) for key in fields},
                            discovery_version=SUPPORTED_VERSIONS.index(configuration['version']) + 1)
        except KeyError as error:
            raise ValueError("Incomplete evidence candidate configuration") from error
        if generator.configuration() != configuration:
            raise ValueError("Evidence candidate configuration does not reconstruct exactly")
        return generator

    def _followups(self, public):
        projection = public.get('evidence')
        if not isinstance(projection, Mapping):
            return []
        if projection.get('match_id') != self.identity['match_id']:
            raise ValueError("Follow-up evidence belongs to another match")
        if (self._discovery_version >= 5
                and projection.get('schema_version') != _FOLLOWUP_VERSIONS[self._discovery_version]):
            return []
        documents = {doc.get('document_id') for doc in public.get('documents', []) if doc.get('status') == 'OK'}
        evidence_ids = {record.get('evidence_id') for record in projection.get('evidence', [])
                        if record.get('anchor', {}).get('document_id') in documents}
        queries = set()
        for proposal in projection.get('state', {}).get('followup_candidates', []):
            ids = proposal.get('based_on_evidence_ids')
            query = proposal.get('query')
            if (proposal.get('tool') != 'search_web' or proposal.get('expected_gain') is not None
                    or proposal.get('proposal_status') != 'grounded_query_not_value_ranked'
                    or not isinstance(ids, list) or not ids or any(key not in evidence_ids for key in ids)
                    or not isinstance(query, str) or not query.strip() or len(query) > 1024):
                continue
            queries.add(query.strip())
        return sorted(queries)[:self.max_followups]

    def _links(self, public, *, bounded=True):
        found = {}
        # Successful admitted documents only. A failure body cannot expand the
        # discovery menu even if a caller attaches links to an error response.
        for document in public.get('documents', []):
            if document.get('status') != 'OK':
                continue
            if self._discovery_version >= 12 and not _visible_parent(document, public):
                continue
            try:
                source_url = _url(document['url'], self.allowed_hosts)
            except (KeyError, TypeError, ValueError):
                continue
            source_entry = source_url
            if self._discovery_version >= 5:
                # The actor still has to obtain this document and observe its
                # links. A registered official host grants link capability,
                # never article truth, relevance or positive information value.
                source_entry = self._source_entry_by_host.get(urlsplit(source_url).hostname, source_url)
            official_host = self._club_entry_hosts.get(source_entry)
            for link in document.get('links', []):
                try:
                    candidate = _url(link['url'], self.allowed_hosts)
                except (KeyError, TypeError, ValueError):
                    continue
                title = str(link.get('title', ''))
                same_club = official_host and urlsplit(candidate).hostname == official_host
                route = self._local_article_routes.get(source_entry)
                path = unquote(urlsplit(candidate).path).casefold()
                slug = path.strip('/').rsplit('/', 1)[-1].removeprefix('muzi-a-')
                local_article = (same_club and route and re.fullmatch(route, path)
                                 and slug not in _INDEX_PARTS)
                official_article = (urlsplit(candidate).hostname == self._official_article_hosts.get(source_entry)
                    and re.fullmatch(r'/[a-z0-9][a-z0-9-]+/?', path)
                    and slug not in _INDEX_PARTS and not _NON_ARTICLE.search(path + ' ' + title))
                dated_article = (self._discovery_version >= 8 and same_club
                    and urlsplit(source_url).hostname == official_host and _dated_article_url(candidate, title))
                finnish_dated_article = (self._discovery_version >= 10 and same_club
                    and source_entry in self._finnish_dated_entries and _visible_parent(document, public)
                    and _finnish_dated_article_url(candidate, title))
                if not _article_url(candidate, title) and not local_article and not official_article and not dated_article and not finnish_dated_article:
                    continue
                haystack = (title + ' ' + unquote(candidate)).casefold().replace('-', ' ').replace('_', ' ')
                named = any(re.search(r'(?<!\w)' + re.escape(team.casefold()) + r'(?!\w)', haystack)
                            for team in self.match_terms)
                if same_club or named or official_article:
                    found.setdefault(candidate, None)
        if self._discovery_version == 1:
            # Preserve exact menus of already collected v1 runs. Never silently
            # apply a later source-order policy to a frozen acquisition trace.
            return sorted(set(found) - set(self.seed_urls))[:self.max_links]
        # Version 2 preserves the archived source order under the finite cap.
        links = [url for url in found if url not in self.seed_urls]
        return links[:self.max_links] if bounded else links

    def _visible_order(self, public, actions):
        """Order the existing bounded menu using observed metadata, never bodies.

        Caps apply before this step, so a later attractive headline cannot
        silently displace an earlier source candidate. The sidecar makes that
        possible cap shortage visible without inventing another capability.
        """
        from football_research.acquisition.state import parse_time

        all_links = self._links(public, bounded=False)
        source_ranks = {url: index + 1 for index, url in enumerate(all_links)}
        official_entries = {}
        news_entries = {}
        for entry in self.profile.get('entries', []):
            if entry.get('kind') not in {'club_home', 'news_index'}:
                continue
            url = _url(entry['url'], self.allowed_hosts)
            if url not in self.seed_urls or entry.get('side') not in {'home', 'away'}:
                continue
            official_entries.setdefault(urlsplit(url).hostname, set()).add(entry['side'])
            if entry['kind'] == 'news_index':
                news_entries.setdefault(url, set()).add(entry['side'])
        observations, visited = {}, set()
        for document_index, document in enumerate(public.get('documents', [])):
            if document.get('status') != 'OK' or not _visible_parent(document, public):
                continue
            try:
                source_url = _url(document['url'], self.allowed_hosts)
            except (KeyError, TypeError, ValueError):
                continue
            visited.add(source_url)
            parent_host = urlsplit(source_url).hostname
            for link_index, link in enumerate(document.get('links', [])):
                try:
                    url = _url(link['url'], self.allowed_hosts)
                except (KeyError, TypeError, ValueError):
                    continue
                if url not in source_ranks:
                    continue
                # An explicit label list was already visibility-filtered by
                # the projector. Empty labels must not resurrect older titles.
                labels = link.get('observed_anchors')
                text = (' '.join(item.get('text', '') for item in labels
                                if isinstance(item, Mapping) and isinstance(item.get('text'), str))
                        if isinstance(labels, list) else str(link.get('title', '')))
                if (self._discovery_version >= 13
                        and link.get('directory_context_basis') in {'liverpool_visible_article_card_v1', 'leeds_visible_contentful_card_v1'}
                        and link.get('target_article_fetched_for_context') is False):
                    text += ' ' + str(link.get('title', ''))
                if (self._discovery_version >= 14
                        and link.get('directory_context_basis') == 'same_response_cms_article_card_v1'):
                    from .candidate_context import _labels
                    cms_labels, _, _ = _labels(link)
                    text += ' ' + ' '.join(item['text'] for item in cms_labels)
                snippet = link.get('snippet')
                if isinstance(snippet, str):
                    text += ' ' + snippet
                same_official = parent_host in official_entries and urlsplit(url).hostname == parent_host
                if url in observations and (observations[url]['official_same_host'] or not same_official):
                    continue
                sides = official_entries.get(parent_host, set()) if same_official else set()
                # A shared competition host has no unique club side. Its
                # metadata cannot establish which team is an "opponent".
                opponents = ({other for side in sides for other in ('home', 'away') if side != other}
                             if len(sides) == 1 else set())
                aliases = [alias for side in opponents for alias in self.identity['aliases'][side]]
                opponent = any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text, re.I)
                               for alias in aliases)
                kickoff_date = parse_time(self.identity['kickoff_at']).date()
                month = ('January February March April May June July August September October November December'
                         .split()[kickoff_date.month - 1])
                full_dates = [kickoff_date.isoformat(), f'{kickoff_date.day} {month} {kickoff_date.year}',
                              f'{month} {kickoff_date.day}, {kickoff_date.year}']
                dated = any(re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', text, re.I)
                            for value in full_dates)
                observations[url] = {
                    'source_rank': source_ranks[url], 'parent_document_id': document.get('document_id'),
                    'document_index': document_index, 'link_index': link_index,
                    'official_same_host': bool(same_official),
                    'directory_route_role': _landing_page_role_v2({'url': url}),
                    'cues': {'opponent_and_full_fixture_date': bool(opponent and dated),
                             'injury_or_training': bool(_INJURY_CUE.search(text)),
                             'opponent': bool(opponent), 'full_fixture_date': bool(dated)},
                }
        attempted = {row.get('information_key') for row in public.get('attempts', [])}
        rows = {}
        keyed = []
        for menu_rank, action in enumerate(actions):
            url = action.get('url')
            observation = observations.get(url)
            unread_directory = (url in news_entries and url not in visited
                                and action['information_key'] not in attempted)
            if action.get('tool') not in {'read_url', 'search_web'}:
                tier = 0
            elif observation and observation['official_same_host'] and not observation['directory_route_role']:
                tier = 1
            elif unread_directory:
                tier = 2 if 'away' in news_entries[url] else 3
            elif url in self.seed_urls:
                tier = 4
            elif action.get('tool') == 'read_url':
                tier = 5
            else:
                tier = 6
            cues = observation['cues'] if observation else {}
            cue_order = tuple(-int(cues.get(key, False)) for key in _VISIBLE_ORDER['article_cues']) if tier == 1 else ()
            source_rank = observation['source_rank'] if observation else None
            key = (tier, cue_order, source_rank if source_rank is not None else menu_rank)
            keyed.append((key, action))
            rows[action['action_id']] = {
                'tier': _VISIBLE_ORDER['order'][tier], 'source_rank': source_rank,
                'source_observation': deepcopy(observation),
                'selected_or_value_estimated_here': False,
            }
        ordered = [action for _, action in sorted(keyed, key=lambda item: item[0])]
        included_urls = {action.get('url') for action in actions}
        cap = {
            'max_discovered_links': self.max_links, 'eligible_observed_article_links': len(all_links),
            'retained_by_link_cap': min(self.max_links, len(all_links)),
            'omitted_by_link_cap': max(0, len(all_links) - self.max_links),
            'omitted_article_links_by_menu_cap': sum(url not in included_urls for url in all_links[:self.max_links]),
            'max_candidates': self.max_candidates,
            'global_candidate_generation_complete': False,
        }
        cap['article_cap_shortage'] = bool(cap['omitted_by_link_cap'] or cap['omitted_article_links_by_menu_cap'])
        return ordered, {'version': _VISIBLE_ORDER['version'], 'by_action_id': rows, 'caps': cap,
                         'ordered_action_ids': [action['action_id'] for action in ordered],
                         'semantics': _VISIBLE_ORDER['semantics']}

    def __call__(self, public):
        kickoff = public['state']['base']['kickoff']
        if kickoff[:10] != self.identity['kickoff_at'][:10]:
            raise ValueError("Public state differs from configured fixture date")
        generic = ' '.join(self.match_terms) + ' ' + kickoff[:10] + ' match preview'
        proposed = [_web_action('search_web', generic, cost=self.web_cost)]
        proposed.extend(_web_action('read_url', url, cost=self.web_cost) for url in self.seed_urls)
        proposed.extend(_web_action('search_web', query, cost=self.web_cost) for query in self._followups(public))
        proposed.extend(_web_action('read_url', url, cost=self.web_cost) for url in self._links(public))
        actions = deepcopy(self.numeric_actions)
        used = {action['information_key'] for action in actions}
        for action in proposed:
            if len(actions) >= self.max_candidates:
                break
            if action['information_key'] not in used:
                actions.append(action)
                used.add(action['information_key'])
        return self._visible_order(public, actions)[0] if self._discovery_version >= 12 else actions


__all__ = ['VERSION', 'LEGACY_VERSION', 'LOCAL_PATH_VERSION', 'OFFICIAL_SUBDOMAIN_VERSION',
           'CONTENT_ROLE_VERSION', 'HTML_BODY_VERSION', 'OBSERVED_CONTEXT_VERSION', 'DATED_ARTICLE_VERSION',
           'FINNISH_DATED_ARTICLE_VERSION', 'VISIBLE_ORDER_VERSION',
           'SUPPORTED_VERSIONS', 'EvidenceDocumentCandidates']
