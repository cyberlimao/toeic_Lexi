"""Portable bindings for explicitly scoped Part 5 editorial evidence.

The old complete critic response stays unchanged. It is NOT a response to the
new complete unit: revised explanation fields are certified only by the current
independent teaching audit. File/API evidence is revalidated by the runner and
release report builder, rather than embedded as fictitious new model output.
"""
from __future__ import annotations
import copy

from .independent_audits import TEACHING_FIELDS, digest
from .models import Unit, blind_payload

PROTOCOL = 'part5-scoped-editorial-evidence-v1'


def non_explanation_projection(unit: Unit):
    value = copy.deepcopy(unit.model_dump())
    for question in value['questions']:
        question.pop('explanation')
    return value


def explanation_fields(unit: Unit):
    rows = {}
    for question in unit.questions:
        exp = question.explanation.model_dump()
        values = {field: exp[field] for field in ('summary_zh', 'reasoning_zh', 'evidence', 'takeaway_zh')}
        values.update({row['label']: row['reason_zh'] for row in exp['option_analysis']})
        rows.update({(question.id, field): values[field] for field in TEACHING_FIELDS})
    return rows


def explanation_change_bindings(before: Unit, after: Unit):
    if before.part != 5 or after.part != 5 or len(before.questions) != 30 or len(after.questions) != 30:
        raise ValueError('Scoped editorial reuse is limited to a complete 30-question Part 5 unit')
    if non_explanation_projection(before) != non_explanation_projection(after):
        raise ValueError('Student/key/metadata/construct fields changed; a full current critic is required')
    # Option labels/order are not learner explanatory prose and cannot change.
    for old, new in zip(before.questions, after.questions):
        if [x.label for x in old.explanation.option_analysis] != [x.label for x in new.explanation.option_analysis]:
            raise ValueError('Explanation option labels/order changed')
    old_fields, new_fields = explanation_fields(before), explanation_fields(after)
    changed, unchanged = [], []
    for (qid, field), value in old_fields.items():
        row = {'question_id': qid, 'field': field}
        if value == new_fields[(qid, field)]:
            unchanged.append({**row, 'sha256': digest(value)})
        else:
            changed.append({**row, 'before_sha256': digest(value), 'after_sha256': digest(new_fields[(qid, field)])})
    if not changed:
        raise ValueError('No explanation change; use ordinary exact-request review replay')
    return changed, unchanged


def summary_errors(entry):
    scope = entry.get('provenance', {}).get('scoped_editorial_evidence')
    if scope is None:
        return []
    unit = Unit.model_validate(entry['content'])
    errors = []
    if scope.get('protocol') != PROTOCOL or unit.part != 5 or len(unit.questions) != 30 or unit.documents:
        return ['Unsupported scoped editorial evidence protocol or unit']
    expected = {'current_content_sha256': digest(unit.model_dump()),
                'student_input_sha256': digest(blind_payload(unit)),
                'non_explanation_sha256': digest(non_explanation_projection(unit)),
                'original_editorial_review_sha256': digest(entry['quality']['editorial_review'])}
    for key, value in expected.items():
        if scope.get(key) != value:
            errors.append(f'Scoped editorial {key} binding mismatch')
    if not scope.get('original_content_sha256') or scope['original_content_sha256'] == expected['current_content_sha256']:
        errors.append('Scoped editorial origin must identify the distinct original complete unit')
    if scope.get('original_critic_applies_to_changed_explanations') is not False:
        errors.append('Old critic must not be represented as checking revised explanations')
    if scope.get('fresh_full_external_review_required') is not True:
        errors.append('Scoped editorial evidence requires final full external review')
    if not scope.get('certificate') or not scope.get('certificate_sha256'):
        errors.append('Scoped editorial API certificate is missing')
    fields = explanation_fields(unit)
    observed = []
    for row in scope.get('changed_fields', []):
        key = row.get('question_id'), row.get('field')
        observed.append(key)
        if key not in fields or row.get('after_sha256') != digest(fields[key]) or row.get('before_sha256') == row.get('after_sha256'):
            errors.append('Changed explanation field does not match its scoped binding')
    for row in scope.get('unchanged_explanation_fields', []):
        key = row.get('question_id'), row.get('field')
        observed.append(key)
        if key not in fields or row.get('sha256') != digest(fields[key]):
            errors.append('Unchanged explanation field does not match its scoped binding')
    if not scope.get('changed_fields') or len(observed) != len(set(observed)) or set(observed) != set(fields):
        errors.append('Scoped explanation fields must cover all 30 x 8 fields exactly once')
    return errors
