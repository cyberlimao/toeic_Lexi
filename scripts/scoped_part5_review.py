"""Certify explanation-only Part 5 repairs with explicitly scoped evidence.

This is NOT exact-input replay of a whole critic on the new unit. The original
complete critic is retained solely for unchanged material, while all 30 current
explanations pass independent Claude teaching checks. Known later substantive
explanation findings must be repaired. Disclosed distractor-score disagreements
require unchanged full questions already checked by an Astra/high examiner.
"""
from __future__ import annotations
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

from toeic_engine.anthropic_provider import ANTHROPIC_MODEL, EFFORT, anthropic_schema
from toeic_engine.cli import export
from toeic_engine.engine import Engine, content_hash, critic_payload, prompt, prompt_version, save_json, verify_entry
from toeic_engine.independent_audits import (PROTOCOL as AUDIT_PROTOCOL, digest, explanation_payload,
    run_teaching, teaching_schema, validate_independent)
from toeic_engine.models import (BlindSolution, Challenge, ExplanationRevision, Review, Unit, blind_payload,
    validate_blind, validate_challenge, validate_review, validate_unit)
from toeic_engine.novelty import History
from toeic_engine.scoped_editorial import (PROTOCOL, explanation_change_bindings, explanation_fields,
    non_explanation_projection, summary_errors)
from scripts.resume_api_review import ExactClaudeReplay, file_hash, validate_reuse_event, wire_request
from scripts.resume_api_review import SCHEMAS

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def unit_file(path):
    value = read(path)
    return Unit.model_validate(value.get('content', value))


def review_event(run, role, expected_result, *, payload=None, input_sha256=None):
    """Bind an actual returned event, not a locally filled review sidecar."""
    for path in sorted((Path(run)/'replay-runs').glob('*/events.json')):
        for index, event in enumerate(read(path)):
            if event.get('role') != role or event.get('validation') != 'complete_original_response':
                continue
            if event.get('kind') not in ('api_call', 'exact_request_reuse'):
                continue
            wire_payload = json.loads(event['requested_identity']['wire_request']['messages'][0]['content'])
            if payload is not None and wire_payload != payload:
                continue
            if input_sha256 is not None and digest(wire_payload) != input_sha256:
                continue
            evidence = validate_reuse_event(event)
            schema = teaching_schema([item['question_id'] for item in expected_result['items']]) if role == 'teaching_reviewer' else SCHEMAS[role]
            if schema.model_validate_json(evidence['raw_response']).model_dump() != expected_result:
                continue
            return {'event_file': str(path.resolve()), 'event_index': index, 'event_sha256': digest(event),
                    'request_input_sha256': digest(wire_payload), 'source': copy.deepcopy(event['source'])}
    raise ValueError(f'{run}: no complete actual {role} API event binds the supplied review/input')


def validate_event_reference(ref):
    rows = read(ref['event_file'])
    index = ref['event_index']
    require(isinstance(index, int) and 0 <= index < len(rows), 'Invalid scoped source event index')
    event = rows[index]
    require(digest(event) == ref['event_sha256'] and event['source'] == ref['source'], 'Scoped source event changed')
    evidence = validate_reuse_event(event)
    payload = json.loads(evidence['wire_request']['messages'][0]['content'])
    require(digest(payload) == ref['request_input_sha256'], 'Scoped source payload changed')
    return event, payload, evidence['raw_response']


def editor_step(run, before: Unit, after: Unit, question_ids):
    """Reconstruct a real explanation-editor response, retaining its API proof."""
    candidates = []
    for request_path in Path(run).rglob('*.request.json'):
        request = read(request_path)
        payload = request.get('payload', {})
        if (request.get('system') != prompt('explanation_editor') or payload.get('rejected_draft') != before.model_dump()
                or payload.get('repair_question_ids') != question_ids):
            continue
        response_path = request_path.with_name(request_path.name.replace('.request.json', '.response.txt'))
        usage_path = request_path.parent/'usage.json'
        if not response_path.exists() or not usage_path.exists():
            continue
        require(request.get('provider') == 'anthropic' and request.get('model') == ANTHROPIC_MODEL
                and request.get('effort') == EFFORT, 'Editor API identity mismatch')
        wire = request['wire_request']
        require(request.get('schema') == anthropic_schema(ExplanationRevision)
                and wire == wire_request(prompt('explanation_editor'), payload, ExplanationRevision, wire['max_tokens']),
                'Editor request wire/schema mismatch')
        replacement = ExplanationRevision.model_validate_json(response_path.read_text())
        require([r.question_id for r in replacement.replacements] == question_ids, 'Editor returned wrong question coverage')
        rebuilt = before.model_copy(deep=True)
        changes = {row.question_id: row.explanation for row in replacement.replacements}
        for question in rebuilt.questions:
            if question.id in changes:
                question.explanation = changes[question.id]
        if rebuilt.model_dump() != after.model_dump():
            continue
        for index, row in enumerate(read(usage_path)):
            if (row.get('label') == request_path.name.removesuffix('.request.json')
                    and (row.get('model'), row.get('model_version'), row.get('effort'), row.get('finish_reason'), row.get('stream_complete'))
                    == (ANTHROPIC_MODEL, ANTHROPIC_MODEL, EFFORT, 'end_turn', True)
                    and row.get('content_sha256') == file_hash(response_path)):
                candidates.append((response_path.stat().st_mtime_ns, str(request_path), {
                    'request': artifact(request_path), 'response': artifact(response_path),
                    'usage': str(usage_path.resolve()), 'usage_row_index': index, 'usage_row_sha256': digest(row),
                    'question_ids': question_ids, 'source_content_sha256': content_hash(before),
                    'result_content_sha256': content_hash(after)}))
    require(candidates, f'{run}: no actual Claude explanation revision reconstructs the exact candidate')
    return sorted(candidates, key=lambda row: row[:2])[0][2]


def configs_equal(baseline_run, known_run, candidate_run):
    configs = [read(Path(run)/'run.json') for run in (baseline_run, known_run, candidate_run)]
    for config in configs:
        require(config.get('author_provider') == config.get('review_provider') == 'claude', 'All source roles must use Claude API')
        require(config.get('prompt_version') == prompt_version(), 'Scoped evidence requires the unchanged frozen prompts')
        require(len(config['plans']) == 1 and config['plans'][0]['part'] == 5, 'Only one Part 5 unit is supported')
    require(configs[0]['plans'] == configs[1]['plans'] == configs[2]['plans'], 'Blueprint changed across scoped sources')
    require(configs[0]['history'] == configs[1]['history'] == configs[2]['history'], 'Novelty lineage changed')
    return configs


def verify_astra_calibrations(spec, candidate: Unit, source_review: Review, known_review: Review):
    qids = spec.get('question_ids', [])
    require(qids and len(qids) == len(set(qids)), 'Explicit calibration question IDs are required')
    candidate_questions = {q.id: q for q in candidate.questions}
    source_items = {row.question_id: row for row in source_review.item_reviews}
    known_items = {row.question_id: row for row in known_review.item_reviews}
    proofs = []
    for external in spec.get('external_reviews', []):
        unit = unit_file(external['unit'])
        review = read(external['review'])
        require((review.get('model'), review.get('reasoning_effort')) == ('gpt-6-astra', 'high'), 'Calibration evidence must come from Astra/high')
        require(review.get('content_sha256') == content_hash(unit), 'Calibration Astra review content binding mismatch')
        questions = {q.id: q for q in unit.questions}
        rows = {row['question_id']: row for row in review['items']}
        for qid in qids:
            require(qid in questions and questions[qid].model_dump() == candidate_questions[qid].model_dump(),
                    f'{qid}: calibration evidence does not cover the exact unchanged full question')
            row = rows[qid]
            require(row.get('verdict') == 'pass' and row.get('independent_answer') == row.get('correct_answer') == questions[qid].answer,
                    f'{qid}: prior Astra item did not pass')
            require([r.get('label') for r in row['option_checks']] == list('ABCD')
                    and all(r.get('verdict') == 'pass' and r.get('comment') for r in row['option_checks']),
                    f'{qid}: prior Astra explanation checks are incomplete')
        proofs.append({'unit': artifact(external['unit']), 'review': artifact(external['review'])})
    require(proofs, 'Calibration disagreement needs independently checked unchanged questions')
    rows = []
    for qid in qids:
        previous, later = source_items[qid], known_items[qid]
        require(previous.distractor_quality >= 4 and later.distractor_quality < 4,
                f'{qid}: not a documented distractor-quality calibration disagreement')
        require(later.explanation_accuracy >= 4 and all(o.explanation_supported for o in later.option_checks),
                f'{qid}: cannot classify a teaching error as calibration disagreement')
        reason = spec.get('reasons', {}).get(qid, '')
        require(len(reason) >= 30, f'{qid}: a concrete retained disagreement reason is required')
        rows.append({'question_id': qid, 'classification': 'calibration_disagreement',
                     'original_score': previous.distractor_quality, 'later_score': later.distractor_quality,
                     'reason': reason, 'fresh_full_astra_review_required': True})
    return rows, proofs


def build_scope(baseline_run: Path, known_run: Path, candidate_run: Path, calibration_path: Path, expected_fields):
    baseline_run, known_run, candidate_run = map(lambda p: Path(p).resolve(), (baseline_run, known_run, candidate_run))
    configs = configs_equal(baseline_run, known_run, candidate_run)
    baseline = unit_file(baseline_run/'attempts/u001-a1.json')
    known = unit_file(known_run/'attempts/u001-a1.json')
    candidate = Unit.model_validate(configs[2]['initial_draft'])
    changed, unchanged = explanation_change_bindings(baseline, candidate)
    actual_fields = {(r['question_id'], r['field']) for r in changed}
    require(actual_fields == set(map(tuple, expected_fields)), 'Candidate differs outside the explicitly declared explanation fields')
    require(non_explanation_projection(known) == non_explanation_projection(baseline), 'Known later review changed student material or metadata')
    for unit in (baseline, known, candidate):
        require(not validate_unit(unit, configs[0]['plans'][0]), 'Original or candidate fails shape/evidence/blueprint validation')
    source_proofs = {}
    review_objects = []
    for name, run, unit in [('baseline', baseline_run, baseline), ('known_later', known_run, known)]:
        solved = BlindSolution.model_validate(read(run/'attempts/u001-a1.blind.json'))
        challenged = Challenge.model_validate(read(run/'attempts/u001-a1.challenge.json'))
        reviewed = Review.model_validate(read(run/'attempts/u001-a1.review.json'))
        require(not validate_blind(solved, unit) and not validate_challenge(challenged, unit), f'{name}: student/key/option defect prevents scoped reuse')
        source_proofs[name] = {
            'solver': review_event(run, 'solver', solved.model_dump(), payload=blind_payload(unit)),
            'challenger': review_event(run, 'challenger', challenged.model_dump(), payload=blind_payload(unit)),
            'critic': review_event(run, 'critic', reviewed.model_dump(), payload=critic_payload(configs[0]['plans'][0], unit, solved, challenged, []))}
        review_objects.append(reviewed)
    original_review, known_review = review_objects
    require(not validate_review(original_review, baseline), 'Baseline critic has an unresolved substantive defect')
    original_independent = read(baseline_run/'attempts/u001-a1.independent.json')
    failures, failed_qids, teaching_proofs = set(), [], []
    for record in original_independent['teaching_review']:
        batch = teaching_schema(record['question_ids']).model_validate(record['result'])
        ref = review_event(baseline_run, 'teaching_reviewer', batch.model_dump(), input_sha256=record['request_input_sha256'])
        _, payload, _ = validate_event_reference(ref)
        require({k:v for k,v in payload.items() if k != 'response_format_correction'} == explanation_payload(baseline, record['question_ids']),
                'Baseline teaching API input is not the original explanation batch')
        teaching_proofs.append(ref)
        for item in batch.items:
            fields = {f.field for f in item.fields if f.verdict != 'pass'} | {i.field for i in item.issues}
            if item.verdict != 'pass' or fields:
                require(fields, 'Unlocalized original teaching failure cannot use scoped repair')
                failures.update((item.question_id, field) for field in fields)
                failed_qids.append(item.question_id)
    require(failures and failures <= actual_fields, 'Every originally failed teaching field must actually change')
    expected_errors = [f'{qid}: teaching review requires correction or resolution' for qid in failed_qids]
    require(validate_independent(original_independent, baseline) == expected_errors, 'Original failure was not limited to the repaired teaching fields')
    rejection = read(baseline_run/'attempts/u001-a1.rejection.json')
    require(rejection == [{'independent_gate': expected_errors, 'independent_audit': original_independent}],
            'Baseline rejection contains unaccounted structural, novelty, student or editorial errors')
    spec = read(calibration_path)
    calibrations, astra_proofs = verify_astra_calibrations(spec, candidate, original_review, known_review)
    calibration_ids = {r['question_id'] for r in calibrations}
    require(known_review.passes(), 'Later review has unresolved global/major findings')
    known_required_fields, expected_later_errors = set(), []
    for item, question in zip(known_review.item_reviews, known.questions):
        require(item.question_id == question.id and item.acceptable_answers == [question.answer]
                and [o.label for o in item.option_checks] == list('ABCD')
                and [o.label for o in item.option_checks if o.defensible] == [question.answer]
                and item.strongest_rival != question.answer, 'Known later review found a student/key/ambiguity defect')
        require(item.explanation_accuracy >= 4, 'Unlocalized low explanation score cannot be scoped away')
        if item.distractor_quality < 4:
            require(item.question_id in calibration_ids, 'Undisposed distractor finding prevents scoped acceptance')
            expected_later_errors.append(f'{item.question_id}: distractors or explanation need substantive revision')
        bad_fields = {o.label for o in item.option_checks if not o.explanation_supported}
        if bad_fields:
            known_required_fields.update((item.question_id, field) for field in bad_fields)
            expected_later_errors.append(f'{item.question_id}: option-level audit found ambiguity or an unsupported teaching claim')
    require(validate_review(known_review, known) == expected_later_errors, 'Known later review has an additional unaccounted defect')
    require(known_required_fields and known_required_fields <= actual_fields, 'Known substantive explanation error was not repaired')
    known_fields, final_fields = explanation_fields(known), explanation_fields(candidate)
    require(all(known_fields[key] != final_fields[key] for key in known_required_fields), 'Later explanation defect remains unchanged')
    first_ids = [q.id for q in known.questions if q.explanation != baseline.questions[int(q.id[1:])-1].explanation]
    second_ids = [q.id for q in candidate.questions if q.explanation != known.questions[int(q.id[1:])-1].explanation]
    editors = [editor_step(known_run, baseline, known, first_ids), editor_step(candidate_run, known, candidate, second_ids)]
    return {
        'protocol': PROTOCOL, 'prompt_version': prompt_version(),
        'inputs': {'baseline_run': str(baseline_run), 'known_run': str(known_run), 'candidate_run': str(candidate_run),
                   'calibration_path': str(Path(calibration_path).resolve()), 'expected_fields': [list(x) for x in sorted(actual_fields)]},
        'original_content_sha256': content_hash(baseline), 'current_content_sha256': content_hash(candidate),
        'student_input_sha256': digest(blind_payload(candidate)), 'non_explanation_sha256': digest(non_explanation_projection(candidate)),
        'original_editorial_review_sha256': digest(original_review.model_dump()),
        'changed_fields': changed, 'unchanged_explanation_fields': unchanged,
        'original_failed_teaching_fields': [list(x) for x in sorted(failures)],
        'later_substantive_explanation_fields': [list(x) for x in sorted(known_required_fields)],
        'calibration_disagreements': calibrations, 'prior_astra_evidence': astra_proofs,
        'source_api_evidence': source_proofs, 'original_teaching_api_evidence': teaching_proofs,
        'actual_explanation_editor_steps': editors,
        'source_artifacts': [artifact(baseline_run/'attempts/u001-a1.json'), artifact(known_run/'attempts/u001-a1.json'),
                             artifact(candidate_run/'run.json'), artifact(calibration_path)],
        'original_critic_applies_to_changed_explanations': False, 'fresh_full_external_review_required': True,
        'scope_note': 'The original complete critic response is retained verbatim as historical evidence for the unchanged student/key/metadata and unchanged explanatory fields. It is not a response to the new complete unit and is not counted as exact-input whole-critic replay or a new API call. Changed explanatory fields require the current independent teaching audit. Known distractor-score disagreements remain disclosed and require fresh full Astra adjudication.'}


def scope_summary(scope, certificate_path):
    fields = ('protocol', 'original_content_sha256', 'current_content_sha256', 'student_input_sha256',
              'non_explanation_sha256', 'original_editorial_review_sha256', 'changed_fields',
              'unchanged_explanation_fields', 'original_critic_applies_to_changed_explanations',
              'fresh_full_external_review_required', 'scope_note')
    return {**{k:scope[k] for k in fields}, 'certificate': str(Path(certificate_path).resolve()),
            'certificate_sha256': file_hash(certificate_path)}


def validate_scoped_entry(entry, source_root=None):
    """Rebuild source proof and current teaching proof at acceptance/publication."""
    summary = entry['provenance']['scoped_editorial_evidence']
    require(not summary_errors(entry), 'Invalid portable scoped editorial bindings')
    certificate_path = Path(summary['certificate'])
    if source_root is not None:
        require(certificate_path.resolve().is_relative_to(Path(source_root).resolve()), 'Scoped certificate outside the reported run')
    require(file_hash(certificate_path) == summary['certificate_sha256'], 'Scoped certificate changed')
    certificate = read(certificate_path)
    scope = build_scope(**{k:Path(v) if k != 'expected_fields' else v for k,v in certificate['scope']['inputs'].items()})
    require(scope == certificate['scope'], 'Scoped source/API evidence differs from the certified inputs')
    require(certificate['status'] == 'accepted_scoped_editorial_and_current_teaching', 'Scoped teaching is incomplete')
    require(scope_summary(scope, certificate_path) == summary, 'Scoped summary differs from the verified certificate')
    unit = Unit.model_validate(entry['content'])
    require(read(Path(scope['inputs']['baseline_run'])/'attempts/u001-a1.review.json') == entry['quality']['editorial_review'],
            'Original critic response was modified')
    final = certificate['final_review_evidence']
    for role, field in [('solver','blind_solution'),('challenger','option_challenge')]:
        _, payload, response = validate_event_reference(final[role])
        require(payload == blind_payload(unit) and SCHEMAS[role].model_validate_json(response).model_dump() == entry['quality'][field], 'Final student review input/output mismatch')
    records = entry['quality']['independent_audit']['teaching_review']
    require(len(final['teaching']) == len(records), 'Final scoped teaching proof is incomplete')
    for ref, record in zip(final['teaching'], records):
        _, payload, response = validate_event_reference(ref)
        require(digest(payload) == record['request_input_sha256']
                and {k:v for k,v in payload.items() if k != 'response_format_correction'} == explanation_payload(unit,record['question_ids'])
                and teaching_schema(record['question_ids']).model_validate_json(response).model_dump() == record['result'],
                'Final teaching request/response differs from current explanation batch')
    require(not validate_independent(entry['quality']['independent_audit'], unit), 'Current independent teaching did not pass')
    return certificate


def run_scoped(baseline_run, known_run, candidate_run, calibration_path, expected_fields, out, source_root, *, max_calls=12, timeout=900):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh scoped-review output directory')
    scope = build_scope(baseline_run, known_run, candidate_run, calibration_path, expected_fields)
    config = copy.deepcopy(read(Path(candidate_run)/'run.json'))
    unit = Unit.model_validate(config['initial_draft'])
    plan = config['plans'][0]
    history = History(Path(config['history']))
    try:
        baseline = unit_file(Path(baseline_run)/'attempts/u001-a1.json')
        require(not history.check(baseline, exclude_source=config.get('revises')), 'Baseline novelty defect prevents scoped reuse')
        require(not history.check(unit, exclude_source=config.get('revises')), 'Candidate fails novelty checks')
        out.mkdir(parents=True)
        config.update(run_id=out.name, created_at=datetime.now(timezone.utc).isoformat(), initial_feedback=[], explanation_only=False)
        config['scoped_editorial_protocol'] = PROTOCOL
        save_json(out/'run.json', config)
        save_json(out/'attempts/u001-a1.json', unit.model_dump())
        original = read(Path(baseline_run)/'attempts/u001-a1.review.json')
        save_json(out/'attempts/u001-a1.review.json', original)
        invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        folder = out/'replay-runs'/invocation
        provider = ExactClaudeReplay(folder, [Path(source_root)], max_calls, timeout)
        engine = Engine(provider, history, out, max_attempts=1, progress=lambda text: print(text, flush=True))
        certificate_path = out/'scoped-editorial-certificate.json'
        certificate = {'status':'pending_current_teaching','scope':scope}
        save_json(certificate_path, certificate)
        try:
            solved = engine._review_call(prompt('solver'), blind_payload(unit), BlindSolution, 'u001-a1-blind', 32768)
            challenged = engine._review_call(prompt('challenger'), blind_payload(unit), Challenge, 'u001-a1-challenge', 49152)
            require(not validate_blind(solved, unit) and not validate_challenge(challenged, unit), 'Current student review failed')
            require(all(e['kind'] == 'exact_request_reuse' for e in provider.events), 'Initial student reviews must use identical existing requests')
            save_json(out/'attempts/u001-a1.blind.json',solved.model_dump())
            save_json(out/'attempts/u001-a1.challenge.json',challenged.model_dump())
            print('Scoped original critic retained; no new or exact-input whole-critic review claimed.',flush=True)
            records = run_teaching(unit, [q.id for q in unit.questions], engine._review_call, prompt('teaching_reviewer'),
                                   'u001-a1', lambda name,row:save_json(out/'attempts'/f'{name}.json',row))
            independent = {'protocol':AUDIT_PROTOCOL,'content_sha256':content_hash(unit),'document_ablation':[],'teaching_review':records}
            save_json(out/'attempts/u001-a1.independent.json',independent)
            errors = validate_independent(independent,unit)
            if errors:
                certificate['status']='rejected_current_teaching'
                save_json(certificate_path,certificate)
                save_json(out/'attempts/u001-a1.rejection.json',[{'independent_gate':errors,'independent_audit':independent}])
                raise ValueError(f'Current teaching failed; no accepted checkpoint: {errors}')
            final_evidence = {
                'solver':review_event(out,'solver',solved.model_dump(),payload=blind_payload(unit)),
                'challenger':review_event(out,'challenger',challenged.model_dump(),payload=blind_payload(unit)),
                'teaching':[review_event(out,'teaching_reviewer',r['result'],input_sha256=r['request_input_sha256']) for r in records]}
            certificate.update(status='accepted_scoped_editorial_and_current_teaching',final_review_evidence=final_evidence)
            save_json(certificate_path,certificate)
            provenance = {**config['initial_author_provenance'],'quality_protocol':AUDIT_PROTOCOL,'prompt_version':prompt_version(),
                          'accepted_at':datetime.now(timezone.utc).isoformat(),'revises':config.get('revises'),
                          'revision_mode':'explanations_only_with_scoped_editorial_evidence',
                          'reviewer':{**provider.review_provenance,'editorial_scope':PROTOCOL},
                          'explanation_editor':provider.author_provenance,
                          'scoped_editorial_evidence':scope_summary(scope,certificate_path),
                          'review_execution':{'mode':provider.index['protocol'],'events':str(folder/'events.json'),'summary':str(folder/'summary.json'),
                              'note':'Only solver/challenger/current teaching use exact-request replay. Historical whole critic is separately scoped evidence, never a replay event for the revised complete unit.'}}
            entry = {'checkpoint_version':2,'id':'u001','content_sha256':content_hash(unit),'blueprint':plan,'content':unit.model_dump(),
                     'quality':{'status':'accepted_automated','attempts':1,'shape_and_evidence':'pass','novelty':'pass',
                        'blind_solution':solved.model_dump(),'option_challenge':challenged.model_dump(),'editorial_review':original,
                        'independent_audit':independent,'external_review':None},'provenance':provenance}
            require(not verify_entry(entry), 'Combined scoped/current gates failed')
            validate_scoped_entry(entry,source_root)
            save_json(out/'units/u001.pending.json',entry)
            history.accept(f'{out}/u001',unit,plan,revises=config.get('revises'))
            (out/'units/u001.pending.json').replace(out/'units/u001.json')
            export(out,config,[entry])
            print(f'Accepted with disclosed scoped editorial evidence: {content_hash(unit)}',flush=True)
        finally:
            provider.close()
            print(json.dumps(read(folder/'summary.json'),ensure_ascii=False),flush=True)
    finally:
        history.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-run','known-run','candidate-run','calibration-path','out','source-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--changed-field',action='append',required=True,help='Exact question:field, e.g. q5:takeaway_zh')
    parser.add_argument('--max-api-calls',type=int,default=12)
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args()
    run_scoped(args.baseline_run,args.known_run,args.candidate_run,args.calibration_path,
               [tuple(x.split(':',1)) for x in args.changed_field],args.out,args.source_root,
               max_calls=args.max_api_calls,timeout=args.timeout)


if __name__=='__main__':
    main()
