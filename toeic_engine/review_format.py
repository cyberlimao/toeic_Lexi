"""Conservative eligibility and immutable-field checks for a metadata retry.

This does not infer a missing document ID or change a review verdict. The
reviewer must supply the IDs in a new request; every other field stays locked.
"""
from __future__ import annotations

import copy
import re

from pydantic import ValidationError

from .models import Review, Unit, validate_review


def part7_missing_document_ids(review: Review, raw_unit: dict) -> list[str]:
    """Return narrowly eligible omissions, never a substantive evidence repair."""
    if raw_unit.get('part') != 7:
        return []
    try:
        unit = Unit.model_validate(raw_unit)
    except (ValidationError, TypeError):
        return []
    by_id = {q.id: q for q in unit.questions}
    missing = [r for r in review.item_reviews if not r.necessary_document_ids]
    if not missing or any(r.question_id not in by_id or by_id[r.question_id].kind == 'cross_document'
                          for r in missing):
        return []
    # All existing substantive judgments and document coverage must already
    # pass. In particular, a one-document cross-document shortcut elsewhere
    # prevents this narrowly scoped format path from running.
    expected = [f'{r.question_id}: invalid/minimal evidence document list' for r in missing]
    if validate_review(review, unit) != expected:
        return []
    targets = {r.question_id for r in missing}
    if any(f.question_id in targets | {'unit'} and f.category in {'evidence', 'ambiguity', 'answer'}
           for f in review.findings):
        return []
    negative = re.compile(
        r'\b(?:no|zero)\s+(?:documents?|passages?|texts?)\b|'
        r'\b(?:without|independent of)\s+(?:any\s+|the\s+)?(?:documents?|passages?|texts?|context)\b|'
        r'\b(?:stem|question|options?|choices?)\b.{0,45}\b(?:alone|only|suffice|sufficient)\b|'
        r'\b(?:world knowledge|common sense|guess(?:ing)?)\b', re.I)
    cross_sentence = re.compile(
        r'\b(?:adjacent|consecutive|neighboring|preceding|following|earlier|later|multiple|two|several)\s+sentences?\b|'
        r'\b(?:requires?|must|depends?|joining|linked|weighing|across)\b', re.I)
    document_terms = {'email': r'e-?mail', 'web_page': r'web\s*page', 'chat': r'(?:chat|conversation|message)',
                      'text_message': r'(?:text\s*message|conversation|message)'}
    known_source = re.compile('|'.join(
        [rf'\b{re.escape(d.id)}(?:-b\d+)?\b' for d in unit.documents]
        + [r'\b' + document_terms.get(d.genre, re.escape(d.genre).replace('_', r'\s+')) + r'\b'
           for d in unit.documents]
        + [r'\b(?:adjacent|consecutive|neighboring|preceding|following|earlier|later|multiple|two|several)\s+sentences?\b']), re.I)
    for item in missing:
        reason = item.context_justification
        if (not item.requires_cross_sentence_context or len(reason) < 20 or negative.search(reason)
                or not cross_sentence.search(reason) or not known_source.search(reason)):
            return []
    return [r.question_id for r in missing]


def document_id_repair_errors(original: Review, revised: Review, allowed: list[str], raw_unit: dict) -> list[str]:
    """Require exact equality outside the named previously empty ID lists."""
    before, after = original.model_dump(), revised.model_dump()
    errors = []
    valid_ids = {d['id'] for d in raw_unit['documents']}
    if [r['question_id'] for r in before['item_reviews']] != [r['question_id'] for r in after['item_reviews']]:
        return ['Locked review question order/coverage changed']
    normalized = copy.deepcopy(after)
    for old, new in zip(before['item_reviews'], normalized['item_reviews']):
        if old['question_id'] in allowed:
            ids = new['necessary_document_ids']
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= valid_ids:
                errors.append(f"{old['question_id']}: repair must supply distinct existing document IDs")
            new['necessary_document_ids'] = old['necessary_document_ids']
    if normalized != before:
        errors.append('Locked review fields changed outside the permitted necessary_document_ids lists')
    return errors
