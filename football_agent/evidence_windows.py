'''Deterministic, exact-offset views over an already acquired source document.

Previews guide navigation and are exposed information, not evidence of unread
body coverage. Offsets are zero-based Python Unicode code points, end exclusive.
'''
from copy import deepcopy
from hashlib import sha256

from .audit import digest

VERSION = 'exact-evidence-windows-v1'


class EvidenceWindows:
    def __init__(self, document, *, window_chars=1200, preview_chars=80):
        if document.get('status') != 'OK' or document.get('kind') != 'page':
            raise ValueError('A successfully acquired page is required')
        text = document.get('text')
        if not isinstance(text, str) or not text:
            raise ValueError('Nonempty extracted text is required')
        if type(window_chars) is not int or not 200 <= window_chars <= 2400:
            raise ValueError('Window length must be 200..2400 code points')
        if type(preview_chars) is not int or not 0 <= preview_chars <= 100:
            raise ValueError('Preview length must be 0..100 code points')
        if len(text) > 24 * window_chars:
            raise ValueError('Document exceeds the explicit 24-window catalog limit')
        self.document = deepcopy(document)
        self.text = text
        self.window_chars, self.preview_chars = window_chars, preview_chars
        self.text_sha256 = sha256(text.encode('utf-8')).hexdigest()
        self.windows = {}
        for start in range(0, len(text), window_chars):
            end = min(start + window_chars, len(text))
            identity = 'w-' + digest({'document_id': document['document_id'],
                'text_sha256': self.text_sha256, 'start': start, 'end': end})[:20]
            self.windows[identity] = {'window_id': identity, 'start': start, 'end': end,
                'text': text[start:end], 'fragment_boundary': True}

    def configuration(self):
        return {'version': VERSION, 'window_chars': self.window_chars,
            'preview_chars': self.preview_chars, 'max_catalog_windows': 24,
            'offset_unit': 'zero_based_unicode_code_points_end_exclusive',
            'ranking': 'source_order_no_relevance_scores', 'new_http_attempts_per_read': 0,
            'reading_cost': 'delivered_characters_and_model_calls_not_monetary_VoI'}

    def manifest(self):
        return {**{k: self.document.get(k) for k in ('document_id', 'url', 'raw_sha256',
            'observed_at', 'available_at', 'source_published_at', 'publication_status')},
            'text_sha256': self.text_sha256, 'extracted_text_chars': len(self.text),
            'extraction_truncated': self.document.get('text_truncated', False),
            'full_source_coverage': 'not_established_by_extracted_text',
            'fresh_observation': False, 'catalog_complete_for_extracted_text': True}

    def catalog(self, opened):
        self._check_opened(opened)
        return [{'window_id': w['window_id'], 'start': w['start'], 'end': w['end'],
            'opened': w['window_id'] in opened,
            'preview': w['text'][:self.preview_chars],
            'preview_start': w['start'],
            'preview_end': w['start'] + min(self.preview_chars, len(w['text'])),
            'preview_scope': 'exposed_fragment_for_navigation_not_full_body'}
            for w in self.windows.values()]

    def _check_opened(self, opened):
        if len(opened) != len(set(opened)) or set(opened) - self.windows.keys():
            raise ValueError('Unknown or duplicate opened window')

    def read(self, window_id):
        if window_id not in self.windows:
            raise ValueError('Unknown window identifier')
        return deepcopy(self.windows[window_id])

    def public(self, opened):
        self._check_opened(opened)
        visible = [self.read(w) for w in opened]
        return {'manifest': self.manifest(), 'catalog': self.catalog(opened),
            'visible_windows': visible,
            'opened_body_chars': sum(w['end'] - w['start'] for w in visible),
            'not_opened_body_ranges': [[w['start'], w['end']] for identity, w in self.windows.items() if identity not in opened],
            'coverage_note': 'Catalog previews were exposed; ranges mark bodies not opened, not wholly unseen text. Unopened text is not absent evidence.'}

    def anchor(self, window_id, quote, opened):
        self._check_opened(opened)
        if window_id not in opened:
            raise ValueError('A source claim requires a previously opened body window')
        window = self.windows[window_id]
        if not isinstance(quote, str) or not 8 <= len(quote) <= 400 or quote not in window['text']:
            raise ValueError('Quote must be an exact 8..400 character substring of the visible body')
        start = window['start'] + window['text'].index(quote)
        return {'document_id': self.document['document_id'], 'window_id': window_id,
            'start': start, 'end': start + len(quote), 'quote': quote,
            'text_sha256': self.text_sha256, 'raw_sha256': self.document['raw_sha256'],
            'occurrence_rule': 'first_exact_match_in_selected_window',
            'semantic_verification': 'quote_location_checked_meaning_unverified'}
