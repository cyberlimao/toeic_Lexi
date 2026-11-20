"""Offline planning and release gates for the built-in-agent diversity workflow.

No model provider is constructed. Native author/reviewer invocations are made by
Codex and recorded in the existing native bridge's actual-invocation ledger.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from toeic_engine.diversity import DiversityHistory, FingerprintAnnotation, content_hash
from toeic_engine.engine import save_json
from toeic_engine.models import Unit
from toeic_engine.novelty import History
from toeic_engine.planning import Planner
from scripts.native_reviewed_test import (ROOT, initialize, read, assignment, reference, native_identity,
    render_native_identity, MODEL, EFFORT, NATIVE_MODELS, NATIVE_EFFORTS, EXTERNAL_EFFORT, EXTERNAL_EFFORTS)

DEFAULT_RUN = ROOT / 'generated/native-reading'

AVOIDANCE = {
    'basis': 'Editorial observations from earlier generated reading exercises and diversity checks.',
    'part5': ['approval/approve derivation family', 'each of + singular finite verb', 'Although versus Despite with interchangeable business scenery', 'explicit by-agent passive as the recurring only voice cue', 'whose + possessed noun as the only relative-pronoun realization', 'weigh A against B', 'twice as many ... as', 'adverb alternatives mixing manner/frequency/degree'],
    'part6': ['One positive sentence versus three unrelated negative choices', 'Local collocation mislabeled as requiring neighboring sentences', 'An operational email every time: include authentic article/review/advertisement voice where planned'],
    'part7': ['Every group opening with purpose', 'Every multiple group centered on a table', 'Four candidates each failing a custom eligibility constraint', 'Correct option alone combining two premises while distractors name one unrelated detail'],
    'limitations': 'These are editorial avoidance priorities for this edition, not permanent bans on valid TOEIC constructs. Archived semantic templates are mostly unknown until explicitly reviewed.'
}


def initialize_diverse(run: Path, seed: int | None, corpus: Path | None = None, run_id: str | None = None,
                       *, model: str = MODEL, effort: str = EFFORT, external_effort: str = EXTERNAL_EFFORT):
    if model not in NATIVE_MODELS or effort not in NATIVE_EFFORTS or external_effort not in EXTERNAL_EFFORTS:
        raise ValueError('Unsupported native model/effort identity')
    if (run/'run.json').exists():
        raise ValueError('Refusing to replace an existing run')
    run.mkdir(parents=True, exist_ok=True)
    history = DiversityHistory(run/'audit/history-before.json')
    sources = history.bootstrap_archives(ROOT/'examples')
    lexical_history = History(run/'novelty.sqlite3')
    try:
        imported_texts = lexical_history.import_corpus(corpus.resolve()) if corpus is not None else 0
    finally:
        lexical_history.close()
    save_json(run/'audit/benchmark-novelty-sources.json', {
        'imported_texts': imported_texts, 'scope': 'Part5 stems and passage text/table rows using production History.import_corpus; source content is not copied into the release.',
        'sources': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in sorted(corpus.glob('test-*/test-*.json'))] if corpus is not None else []})
    planner = Planner(seed, history.usage())
    seed = planner.seed
    plans = planner.build('full')
    save_json(run/'audit/planned-test.json', plans)
    save_json(run/'audit/archive-sources.json', sources)
    save_json(run/'audit/editorial-avoidance.json', AVOIDANCE)
    protocol = (ROOT/'toeic_engine/prompts/native_review_protocol.txt').read_text()
    protocol = protocol.replace('{{WORKSPACE}}', str(ROOT)).replace('{{RUN}}', str(run))
    protocol_file = run/'protocol-input.txt'
    protocol_file.write_text(protocol)
    initialize(run/'audit/planned-test.json', run_id or run.name, seed, protocol_file, run,
               model=model, effort=effort, external_effort=external_effort)
    run_metadata = read(run/'run.json')
    run_metadata['diversity_protocol'] = 'reviewed-fingerprints-and-whole-test-v1'
    whole_instruction = render_native_identity((ROOT/'toeic_engine/prompts/whole_test_reviewer.txt').read_text(),
                                               model, effort, external_effort=external_effort)
    run_metadata['native_instruction_rendering'] = {
        'model_identity': model, 'reasoning_effort': effort, 'source_prompts_unchanged': True,
        'external_review_model': 'gpt-6-astra', 'external_review_effort': external_effort,
        'protocol_role_labels': {'gpt-6.1-terra': 'Terra', 'gpt-6.1-sol': 'Sol', 'gpt-6-astra': 'Astra'}[model],
        'whole_test_comparison_baseline': 'available generated practice editions',
    }
    (run/'instructions/whole_test_reviewer.txt').write_text(whole_instruction)
    run_metadata['diversity_instruction_sha256'] = hashlib.sha256(whole_instruction.encode()).hexdigest()
    save_json(run/'run.json', run_metadata)
    save_json(run/'schemas/fingerprint.schema.json', FingerprintAnnotation.model_json_schema())
    for i, plan in enumerate(plans, 1):
        uid = f'u{i:03d}'
        save_json(run/'author-inputs'/f'{uid}.json', {
            'unit_id': uid, 'blueprint': plan, 'history_avoidance': AVOIDANCE,
            'observed_history': history.planning_context(plan['part'], 12),
            'archive_limitations': sources['limitations'],
            'output_unit': reference(run/'authored'/f'{uid}.json'),
            'output_annotations': reference(run/'annotations'/f'{uid}.author.json'),
            'instructions': reference(run/'instructions/author.txt'),
            'unit_schema': reference(run/'unit.schema.json'),
            'annotation_schema': reference(run/'schemas/fingerprint.schema.json'),
            'annotation_instructions': 'Write one annotation per question in order; use literal normalized keyed option (lowercase English words, retain internal apostrophe/hyphen), specific stable template code, and exact evidence. No reviewer_id until a separate actual critic confirms. All content must be original.'
        })
    print(f'Diversity history, frozen plans and {len(plans)} author packets ready: {run}')


def critic_assignment(run: Path, name: str, units: list[str]):
    assignment(name, 'critic', units, run)
    path = run/'assignments'/f'{name}.json'
    packet = read(path)
    history = DiversityHistory(run/'audit/history-before.json')
    state = read(run/'selected.json')
    for task in packet['tasks']:
        uid = task['unit_id']
        unit = Unit.model_validate(read(ROOT/state[uid]['canonical']))
        author_annotations = run/'annotations'/f'{uid}.author.json'
        raw = read(author_annotations)
        diagnostic = history.audit_unit(unit, read(run/'run.json')['run_id'], uid, raw, archive=True)
        diagnostic['status_note'] = 'Author fingerprint comparison only; this is not reviewer confirmation.'
        target = f"audit/pre-review-diversity-{state[uid]['key']}.json"
        save_json(run/target, diagnostic)
        task['author_annotations'] = str(author_annotations.relative_to(run))
        task['history_diagnostic'] = target
        task['output_annotations'] = f"annotations/{state[uid]['key']}.reviewed.json"
    packet['annotation_schema'] = 'schemas/fingerprint.schema.json'
    packet['annotation_instructions'] = ('Verify every author annotation against actual text and evidence; repair inaccurate classification or citations. '
        'Consider prior matching examples and distinguish legitimate core-skill reuse from cosmetic template reuse. '
        'Write separate {unit_id,content_sha256,annotations:[...]} envelope. Every confirmed annotation needs your actual '
        'reviewer_id and a concrete reviewer_note (20+ characters). Do not confirm semantic claims merely because the author labels them. '
        'Read only these explicitly listed supplemental files plus the original critic inputs; no challenger/teaching/external conclusions.')
    save_json(path, packet)
    print('Added hash-bound fingerprint confirmation and disclosed history diagnostics')


def entries_and_annotations(run):
    selected = read(run/'selected.json')
    entries, annotations = [], {}
    invocations = read(run/'audit/agent-invocations.json')['agents']
    for uid, spec in sorted(selected.items()):
        unit = Unit.model_validate(read(ROOT/spec['canonical']))
        sha = content_hash(unit)
        if spec['content_sha256'] != sha:
            raise ValueError(f'{uid}: stale canonical selection')
        entries.append({'id': uid, 'content': unit.model_dump(), 'content_sha256': sha})
        file = run/'annotations'/f'{uid}-{sha[:12]}.reviewed.json'
        if file.exists():
            packet = read(file)
            if packet.get('unit_id') != uid or packet.get('content_sha256') != sha:
                raise ValueError(f'{uid}: annotation content binding mismatch')
            for a in packet['annotations']:
                actor = a.get('reviewer_id')
                if not any(r['agent'] == actor and r['role'] == 'critic' and uid in r['assigned_units']
                           and (r['model'], r['reasoning_effort']) == native_identity(run) for r in invocations):
                    raise ValueError(f'{uid}: annotation lacks actual independent native critic provenance')
            annotations[uid] = packet['annotations']
    return entries, annotations


def whole_test_metrics(entries):
    units = [Unit.model_validate(e['content']) for e in entries]
    p7 = [u for u in units if u.part == 7]
    multi = [u for u in p7 if len(u.documents) > 1]
    counts = Counter(q.kind for u in p7 for q in u.questions)
    return {'question_counts': {str(p): sum(len(u.questions) for u in units if u.part == p) for p in (5,6,7)},
            'p7_kind_counts': dict(counts), 'p7_first_kind_counts': dict(Counter(u.questions[0].kind for u in p7)),
            'p7_groups': len(p7), 'p7_multiple_groups': len(multi),
            'p7_multiple_with_tables': sum(any(b.type == 'table' for d in u.documents for b in d.blocks) for u in multi),
            'document_genres': dict(Counter(d.genre for u in units for d in u.documents))}


def save_accepted_history(history, entries, edition_id, annotations, exceptions, destination):
    """Publish accepted records without changing the frozen pre-run archive."""
    accepted = DiversityHistory()
    accepted.records = copy.deepcopy(history.records)
    for entry in entries:
        accepted.record_unit(Unit.model_validate(entry['content']), edition_id, entry['id'],
                             annotations[entry['id']], exceptions.get(entry['id']))
    accepted.save(destination)


def audit(run: Path, require_pass=False):
    entries, annotations = entries_and_annotations(run)
    history = DiversityHistory(run/'audit/history-before.json')
    exceptions_path = run/'audit/diversity-exceptions.json'
    result = history.audit_test(entries, read(run/'run.json')['run_id'], annotations,
                                read(exceptions_path) if exceptions_path.exists() else {})
    metrics = whole_test_metrics(entries)
    result['whole_test_metrics'] = metrics
    result['policy'] = {'part7_longest_option_alert': .30, 'part7_shortest_option_alert': .30, 'purpose_planning_range': [5,8], 'multiple_table_group_budget': 3,
                        'status': 'Editorial engineering alerts; not validated official TOEIC quotas or psychometric thresholds.'}
    alerts = []
    run_plans = read(run/'run.json')['plans']
    for entry in entries:
        plan = run_plans[int(entry['id'][1:])-1]
        policy = plan.get('table_policy')
        if policy:
            actual_tables = sum(b['type'] == 'table' for d in entry['content']['documents'] for b in d['blocks'])
            if actual_tables > policy['max_table_blocks']:
                alerts.append(f"{entry['id']}: {actual_tables} table blocks exceed planned maximum {policy['max_table_blocks']}")
    length = result['option_length']['parts']['7']['longest_strategy_expected_accuracy']
    if length is not None and length > .30: alerts.append('Part 7 longest-option strategy exceeds editorial alert .30')
    shortest = result['option_length']['parts']['7']['shortest_strategy_expected_accuracy']
    if shortest is not None and shortest > .30: alerts.append('Part 7 shortest-option strategy exceeds editorial alert .30')
    if metrics['p7_multiple_with_tables'] > 3: alerts.append('More than three multiple groups use tables')
    if metrics['p7_groups'] == 15 and not 5 <= metrics['p7_kind_counts'].get('purpose', 0) <= 8:
        alerts.append('Purpose count outside declared planning range')
    result['editorial_alerts'] = alerts
    if require_pass:
        if hashlib.sha256((run/'instructions/whole_test_reviewer.txt').read_bytes()).hexdigest() != read(run/'run.json')['diversity_instruction_sha256']:
            raise ValueError('Whole-test instructions changed after freezing')
        if len(entries) != 20 or metrics['question_counts'] != {'5':30,'6':16,'7':54}:
            raise ValueError('Final diversity gate requires the complete 100-question test')
        review = read(run/'audit/whole-test-review.json')
        expected = {e['id']: e['content_sha256'] for e in entries}
        actual_actor = any(a['agent'] == review.get('agent') and a['role'] == 'whole_test' and set(a['assigned_units']) == set(expected)
                           and (a['model'],a['reasoning_effort']) == native_identity(run) for a in read(run/'audit/agent-invocations.json')['agents'])
        if (review.get('content_hashes') != expected or review.get('verdict') != 'pass' or not actual_actor
                or (review.get('model'), review.get('reasoning_effort')) != native_identity(run)
                or review.get('unresolved_issues')):
            raise ValueError('Missing passing exact-version native whole-test review')
        if alerts and set(review.get('resolved_alerts', {})) != set(alerts):
            raise ValueError('Whole-test alerts require explicit editorial disposition')
        result['whole_test_review'] = review
    save_json(run/'audit/diversity-audit.json', result)
    print(json.dumps({'units':len(entries), 'annotation_units':len(annotations), 'review_required':result['review_required'],
                      'editorial_alerts':alerts, 'metrics':metrics, 'p7_longest_strategy':length}, ensure_ascii=False))
    if require_pass and result['review_required']:
        raise ValueError('Unresolved diversity findings')
    if require_pass:
        # The pre-run archive is a frozen input. record_unit autosaves when
        # a history has a path, so stage accepted records in an unbound copy.
        save_accepted_history(history, entries, result['edition_id'], annotations,
                              read(exceptions_path) if exceptions_path.exists() else {},
                              run/'audit/history-after.json')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('init');p.add_argument('--seed',type=int,default=None,help='Omit for a new random seed; saved seeds reproduce the exact plans');p.add_argument('--corpus',type=Path,help='Optional external comparison corpus with ten test JSON files');p.add_argument('--run-id',help='Unique edition identity; defaults to the run directory name');p.add_argument('--model',choices=NATIVE_MODELS,default=MODEL)
    p.add_argument('--effort', choices=NATIVE_EFFORTS, default=EFFORT)
    p.add_argument('--external-effort', choices=EXTERNAL_EFFORTS, default=EXTERNAL_EFFORT)
    p=sub.add_parser('audit');p.add_argument('--require-pass',action='store_true')
    p=sub.add_parser('critic-assignment');p.add_argument('name');p.add_argument('units',nargs='+')
    args=parser.parse_args()
    if args.command=='init': initialize_diverse(args.run_dir.resolve(),args.seed,args.corpus,args.run_id,
                                               model=args.model, effort=args.effort, external_effort=args.external_effort)
    elif args.command=='critic-assignment': critic_assignment(args.run_dir.resolve(),args.name,args.units)
    else: audit(args.run_dir.resolve(),args.require_pass)

if __name__=='__main__': main()
