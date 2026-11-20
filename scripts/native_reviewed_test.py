"""File bridge for built-in Codex agents, with the existing engine quality gates.

This module never invokes a model API. Agents write review and provenance records;
Python binds their visible inputs and outputs to hashes and validates coverage.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from toeic_engine.engine import content_hash, save_json, verify_entry, prompt_version, prompt, PROMPT_NAMES
from toeic_engine.models import Unit, blind_payload, validate_unit, BlindSolution, Challenge, Review
from toeic_engine.novelty import History
from toeic_engine.independent_audits import (
    PROTOCOL, digest, document_subsets, ablation_payload, explanation_payload,
    EvidenceDecision, TeachingReview,
    evidence_errors,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL, EFFORT = 'gpt-6.1-sol', 'xhigh'
NATIVE_MODELS = ('gpt-6.1-sol', 'gpt-6.1-terra', 'gpt-6-astra')
NATIVE_EFFORTS = ('medium', 'xhigh')
EXTERNAL_MODEL, EXTERNAL_EFFORT = 'gpt-6-astra', 'high'
EXTERNAL_EFFORTS = ('high', 'xhigh')
RUN = ROOT / 'generated/native-reading'


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def native_identity(run_dir, role='review'):
    """Read the declared native actor identity; never infer an API provider."""
    run = read(Path(run_dir) / 'run.json')
    if role == 'external':
        model = run.get('external_review_model', EXTERNAL_MODEL)
        effort = run.get('external_review_effort', EXTERNAL_EFFORT)
        valid = model == EXTERNAL_MODEL and effort in EXTERNAL_EFFORTS
    else:
        model = run['author_model' if role == 'author' else 'review_model']
        effort = run['reasoning_effort']
        valid = model in NATIVE_MODELS and effort in NATIVE_EFFORTS
    if not valid:
        raise ValueError('Unsupported native model/effort identity')
    return model, effort


def render_native_identity(text, model, effort=EFFORT, *, external_effort=EXTERNAL_EFFORT):
    """Substitute identity labels, leaving substantive role prompts unchanged."""
    if model not in NATIVE_MODELS or effort not in NATIVE_EFFORTS or external_effort not in EXTERNAL_EFFORTS:
        raise ValueError('Unsupported native model/effort identity')
    # Replace the internal identity before the external one. Astra may fill
    # both roles, but their effort levels must remain independently declared.
    rendered = text.replace(f'{MODEL} / {EFFORT}', f'{model} / {effort}')
    rendered = rendered.replace(f'{MODEL}/{EFFORT}', f'{model}/{effort}')
    rendered = rendered.replace(MODEL, model)
    rendered = rendered.replace('"reasoning_effort":"xhigh"', f'"reasoning_effort":"{effort}"')
    rendered = rendered.replace(f'{EXTERNAL_MODEL} / {EXTERNAL_EFFORT}', f'{EXTERNAL_MODEL} / {external_effort}')
    rendered = rendered.replace(f'{EXTERNAL_MODEL}/{EXTERNAL_EFFORT}', f'{EXTERNAL_MODEL}/{external_effort}')
    label = {'gpt-6.1-sol': 'Sol', 'gpt-6.1-terra': 'Terra', 'gpt-6-astra': 'Astra'}[model]
    rendered = rendered.replace('Sol authors', f'{label} authors').replace('Sol critic', f'{label} critic')
    return rendered


def selected_unit(spec):
    unit = Unit.model_validate(read(ROOT / spec['canonical']))
    if content_hash(unit) != spec['content_sha256']:
        raise ValueError('Selected canonical content hash mismatch')
    return unit


def freeze_instructions(run_dir):
    """Freeze actual role inputs; a prompt change requires a new run/version."""
    run_dir = Path(run_dir)
    run = read(run_dir / 'run.json')
    if prompt_version() != run['prompt_version']:
        raise ValueError('Role prompts changed during this native run; declare a new version first')
    prompts = {name: prompt(name) for name in PROMPT_NAMES}
    identity = {'author': native_identity(run_dir, 'author'), 'review': native_identity(run_dir)}
    # Historical snapshots did not declare an external identity. Preserve
    # those snapshots while binding both roles for newly initialized runs.
    if 'external_review_model' in run or 'external_review_effort' in run:
        identity['external'] = native_identity(run_dir, 'external')
    identity = {key: list(value) for key, value in identity.items()}
    protocol = (run_dir / 'native-review-protocol.txt').read_text()
    snapshot_path = run_dir / 'audit/prompt-snapshot.json'
    if snapshot_path.exists():
        snapshot = read(snapshot_path)
        if 'native_identity' in snapshot and snapshot['native_identity'] != identity:
            raise ValueError('Frozen native model identity changed')
        if (snapshot.get('prompt_version') != run['prompt_version']
                or snapshot.get('prompts') != prompts
                or snapshot.get('native_protocol') != protocol):
            raise ValueError('Frozen role/protocol instruction snapshot mismatch')
    else:
        save_json(snapshot_path, {
            'prompt_version': run['prompt_version'], 'prompts': prompts,
            'native_protocol': protocol, 'native_identity': identity,
            'execution': 'Frozen native-agent instructions, not model API messages.',
        })
    for name, text in prompts.items():
        path = run_dir / 'instructions' / f'{name}.txt'
        if path.exists() and path.read_text() != text:
            raise ValueError(f'Frozen instruction changed: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text)


def initialize(plans_path, run_id, seed, protocol_path, run_dir=None, *, model=MODEL,
               effort=EFFORT, external_effort=EXTERNAL_EFFORT):
    """Initialize from saved plans using only local files; never create a provider."""
    run_dir = Path(run_dir) if run_dir is not None else RUN
    if model not in NATIVE_MODELS or effort not in NATIVE_EFFORTS or external_effort not in EXTERNAL_EFFORTS:
        raise ValueError('Unsupported native model/effort identity')
    if (run_dir / 'run.json').exists():
        raise ValueError('Run already exists; refusing to replace its provenance')
    saved = read(plans_path)
    plans = saved['plans'] if isinstance(saved, dict) else saved
    if not isinstance(plans, list) or not plans or any(
            p.get('part') not in (5, 6, 7)
            or not isinstance(p.get('question_count'), int)
            or p['question_count'] < 1 or len(p.get('slots', [])) != p['question_count']
            for p in plans):
        raise ValueError('Saved plans must contain valid nonempty question slots')
    if not run_id.strip():
        raise ValueError('A nonempty run ID is required')
    protocol = render_native_identity(Path(protocol_path).read_text(), model, effort, external_effort=external_effort)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'native-review-protocol.txt').write_text(protocol)
    save_json(run_dir / 'run.json', {
        'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
        'seed': seed, 'plans': plans, 'prompt_version': prompt_version(),
        'author_provider': 'codex_subagent', 'review_provider': 'codex_subagent',
        'author_model': model, 'review_model': model, 'reasoning_effort': effort,
        'external_review_model': EXTERNAL_MODEL, 'external_review_effort': external_effort,
        'execution': 'codex_builtin_subagents', 'native_pipeline_version': 2,
        'model_api_allowed': False, 'provider_instantiation_allowed': False,
    })
    for i, plan in enumerate(plans, 1):
        save_json(run_dir / 'plans' / f'u{i:03d}.json', plan)
    save_json(run_dir / 'selected.json', {})
    save_json(run_dir / 'audit/agent-invocations.json', {
        'execution': 'Root records actual collaboration invocations; no actors are synthesized.',
        'agents': [],
    })
    save_json(run_dir / 'audit/execution.json', {
        'mode': 'codex_builtin_subagents', 'api_calls': 0,
        'model_api_allowed': False, 'provider_instantiation_allowed': False,
        'api_call_count_scope': 'This native run only; all helpers operate on local files.',
    })
    schemas(run_dir)
    print(f'Initialized {run_id}: {len(plans)} saved plans; no agents or providers invoked')


def selected(run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    return read(RUN / 'selected.json') if (RUN / 'selected.json').exists() else {}


def prepare(uid, source, run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    unit = Unit.model_validate(read(source))
    plan = read(RUN / 'plans' / f'{uid}.json')
    run = read(RUN / 'run.json')
    if plan != run['plans'][int(uid[1:])-1]:
        raise ValueError('Unit plan differs from frozen run metadata')
    freeze_instructions(RUN)
    if errors := validate_unit(unit, plan):
        raise ValueError(errors)
    sha = content_hash(unit)
    key = f'{uid}-{sha[:12]}'
    canonical = RUN / 'drafts' / f'{key}.json'
    save_json(canonical, unit.model_dump())
    state = selected(RUN)
    state[uid] = {'source': reference(source),
                  'canonical': reference(canonical), 'content_sha256': sha, 'key': key}
    save_json(RUN / 'selected.json', state)
    start = 101 + sum(p['question_count'] for p in run['plans'][:int(uid[1:])-1])
    common = {'unit_id': uid, 'content_sha256': sha,
              'display_numbers': {q.id: start+i for i, q in enumerate(unit.questions)}}
    save_json(RUN / 'packets' / f'{key}.blind.json', {**common, 'student_material': blind_payload(unit)})
    save_json(RUN / 'packets' / f'{key}.content.json', {**common, 'content': unit.model_dump(), 'blueprint': plan})
    for index in range(0, len(unit.questions), 5):
        ids = [q.id for q in unit.questions[index:index+5]]
        save_json(RUN / 'packets' / f'{key}.teaching-{index//5+1}.json',
                  {**common, 'question_ids': ids, 'payload': explanation_payload(unit, ids)})
    print(key)


def schemas(run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    freeze_instructions(RUN)
    for name, model in [('unit', Unit), ('solver', BlindSolution), ('challenger', Challenge), ('critic', Review),
                        ('teaching', TeachingReview), ('probe', EvidenceDecision)]:
        save_json(RUN / f'{name}.schema.json', model.model_json_schema())
        (RUN / 'reviews' / name).mkdir(parents=True, exist_ok=True)


def assignment(name, role, units, run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    freeze_instructions(RUN)
    tasks = []
    for uid in units:
        spec = selected(RUN)[uid]
        key = spec['key']
        common = {'unit_id': uid, 'content_sha256': spec['content_sha256']}
        if role in ('solver', 'challenger'):
            tasks.append({**common, 'input': f'packets/{key}.blind.json',
                          'output': f'reviews/{role}/{key}.json'})
        elif role == 'critic':
            tasks.append({**common, 'input': f'packets/{key}.content.json',
                          'independent_solver': f'reviews/solver/{key}.json',
                          'output': f'reviews/critic/{key}.json'})
        elif role == 'teaching':
            unit = selected_unit(spec)
            for n in range(1, (len(unit.questions)+4)//5+1):
                out = f'reviews/teaching/{key}-{n}.json'
                if (RUN/out).exists():
                    continue
                tasks.append({**common, 'input': f'packets/{key}.teaching-{n}.json', 'output': out})
        else:
            raise ValueError('Unsupported role')
    model, effort = native_identity(RUN)
    result = {'role': role, 'model': model, 'reasoning_effort': effort,
              'instruction_file': f'instructions/{"teaching_reviewer" if role == "teaching" else role}.txt',
              'protocol_file': 'native-review-protocol.txt', 'prompt_version': read(RUN/'run.json')['prompt_version'],
              'schema_file': f'{role}.schema.json', 'tasks': tasks,
              'boundary': 'Read only the listed inputs and role/protocol/schema instructions; no other role conclusions except explicitly listed independent_solver for critic.'}
    path = RUN/'assignments'/f'{name}.json'
    save_json(path, result)
    print(reference(path), len(tasks), 'review payloads')


def probes(run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    freeze_instructions(RUN)
    groups = {}
    for uid, spec in sorted(selected(RUN).items()):
        unit = selected_unit(spec)
        if len(unit.documents) < 2:
            continue
        for visible in document_subsets(unit):
            mask = '-'.join(visible) or 'none'
            for q in unit.questions:
                if q.kind != 'cross_document':
                    continue
                payload = ablation_payload(unit, q.id, visible)
                groups.setdefault(f'{mask}-{q.id}', []).append({
                    'unit_id': uid, 'content_sha256': spec['content_sha256'],
                    'question_id': q.id, 'visible_document_ids': visible,
                    'input_sha256': digest(payload), 'payload': payload,
                })
    for mask, items in groups.items():
        save_json(RUN / 'packets' / f'probe-{mask}.json', items)
    print({mask: len(items) for mask, items in groups.items()})


def probe_assignment(name, packet_name, run_dir=None):
    run_dir = Path(run_dir) if run_dir is not None else RUN
    freeze_instructions(run_dir)
    packets = read(run_dir / 'packets' / f'probe-{packet_name}.json')
    units = [p['unit_id'] for p in packets]
    if len(set(units)) != len(units):
        raise ValueError('A probe context may see only one question/subset per unit')
    model, effort = native_identity(run_dir)
    value = {'role': 'probe', 'model': model, 'reasoning_effort': effort,
             'input': f'packets/probe-{packet_name}.json',
             'output': f'reviews/probe/{packet_name}.json', 'assigned_units': units,
             'instruction_file': 'instructions/evidence_probe.txt',
             'protocol_file': 'native-review-protocol.txt', 'schema_file': 'probe.schema.json',
             'prompt_version': read(run_dir/'run.json')['prompt_version'],
             'boundary': 'Exactly one question and one fixed visible document subset per unit. Read only this packet, frozen instructions, protocol and schema. Do not read other questions, documents, keys or reviews. A fresh native agent is required for every same-unit question/subset pair.'}
    save_json(run_dir / 'assignments' / f'{name}.json', value)
    print(reference(run_dir / 'assignments' / f'{name}.json'))


def check_probe(name, run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    assignment = read(RUN/'assignments'/f'{name}.json')
    packets, records = read(RUN/assignment['input']), read(RUN/assignment['output'])
    if len(packets) != len(records):
        raise ValueError('Probe coverage mismatch')
    summaries = []
    for packet, record in zip(packets, records):
        for field in ('unit_id', 'content_sha256', 'question_id', 'visible_document_ids', 'input_sha256'):
            if record[field] != packet[field]:
                raise ValueError(f'Probe {field} mismatch')
        spec = selected(RUN)[record['unit_id']]
        if spec['content_sha256'] != record['content_sha256']:
            raise ValueError('Stale probe content')
        unit = selected_unit(spec)
        qid, visible = record['question_id'], record['visible_document_ids']
        validate_record(record, record['unit_id'], spec['content_sha256'], 'probe', RUN)
        payload = ablation_payload(unit, qid, visible)
        if (packet['payload'] != payload or packet['input_sha256'] != digest(payload)
                or visible not in document_subsets(unit)):
            raise ValueError('Probe physical payload/hash mismatch')
        validate_reuse(record, unit, 'probe', RUN, question_id=qid, visible=visible)
        result = EvidenceDecision.model_validate(record['result'])
        errors = evidence_errors(result, unit, qid, visible)
        key = next(q.answer for q in unit.questions if q.id == qid)
        if result.decision == 'uncertain' or any(o.status == 'uncertain' for o in result.options):
            errors.append('Unresolved uncertainty')
        if result.decision == 'sufficient' and result.answer == key and len(visible) < 2:
            errors.append('Zero/one-document shortcut to the complete item key')
        if len(visible) == len(unit.documents) and (result.decision != 'sufficient' or result.answer != key):
            errors.append('Full-evidence positive control failed')
        if errors:
            raise ValueError({record['unit_id']: errors})
        summaries.append({'unit_id': record['unit_id'], 'decision': result.decision, 'answer': result.answer})
    print(name, summaries)


def checked_record(path, uid, sha, role, run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    record = read(path)
    validate_record(record, uid, sha, role, RUN)
    return record


def validate_record(record, uid, sha, role, run_dir):
    path = f'{uid}/{role}'
    model, effort = native_identity(run_dir, role)
    if (record.get('unit_id'), record.get('content_sha256'), record.get('model'), record.get('reasoning_effort')) != (uid, sha, model, effort):
        raise ValueError(f'{path}: identity/model/content mismatch')
    if not record.get('agent') or record.get('role') != role:
        raise ValueError(f'{path}: missing native-agent provenance')
    actors = read(Path(run_dir)/'audit'/'agent-invocations.json')['agents']
    if not any(a['agent'] == record['agent'] and a['role'] == role
               and uid in a.get('assigned_units', [])
               and (a['model'], a['reasoning_effort']) == (model, effort) for a in actors):
        raise ValueError(f'{path}: reviewer not present in the actual invocation ledger')


def validate_reuse(record, unit, role, run_dir, *, question_ids=None, question_id=None, visible=None, seen=None):
    """Recheck saved provenance, exact source result and complete task input."""
    if 'reuse' not in record:
        return
    reuse_record = record['reuse']
    source = ROOT / reuse_record['source']
    seen = set() if seen is None else seen
    if source.resolve() in seen:
        raise ValueError('Cyclic review reuse provenance')
    seen.add(source.resolve())
    original = read(source)
    uid, old_sha = record['unit_id'], reuse_record['source_content_sha256']
    validate_record(original, uid, old_sha, role, run_dir)
    if original['agent'] != record['agent'] or original['result'] != record['result']:
        raise ValueError('Reused review result or actual reviewer was changed')
    old = Unit.model_validate(read(Path(run_dir)/'drafts'/f'{uid}-{old_sha[:12]}.json'))
    if content_hash(old) != old_sha:
        raise ValueError('Reuse source content hash mismatch')
    if role in ('solver', 'challenger'):
        before, after = blind_payload(old), blind_payload(unit)
    elif role == 'teaching':
        before, after = explanation_payload(old, question_ids), explanation_payload(unit, question_ids)
    elif role == 'probe':
        before = ablation_payload(old, question_id, visible)
        after = ablation_payload(unit, question_id, visible)
        if (original.get('question_id') != question_id
                or original.get('visible_document_ids') != visible
                or original.get('input_sha256') != digest(before)):
            raise ValueError('Reused probe source identity/input mismatch')
    else:
        raise ValueError('This role cannot reuse a review')
    if before != after or reuse_record.get('input_sha256') != digest(after):
        raise ValueError('Reused review requires an unchanged complete task payload')
    validate_reuse(original, old, role, run_dir, question_ids=question_ids,
                   question_id=question_id, visible=visible, seen=seen)


def reuse(uid, old_key, run_dir=None):
    """Reuse actual reviews only when their entire question/document/explanation payload is identical."""
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    spec = selected(RUN)[uid]
    old = Unit.model_validate(read(RUN/'drafts'/f'{old_key}.json'))
    new = selected_unit(spec)
    old_sha, sha, key = content_hash(old), spec['content_sha256'], spec['key']
    if old_sha == sha:
        return
    events = []

    def copy_review(role, old_path, new_path, old_input, new_input):
        if old_input != new_input or not old_path.exists() or new_path.exists():
            return
        record = checked_record(old_path, uid, old_sha, role, RUN)
        validate_reuse(record, old, role, RUN,
                       question_ids=[q['id'] for q in old_input['questions']] if role == 'teaching' else None)
        record['content_sha256'] = sha
        record['reuse'] = {'source': reference(old_path),
                           'source_content_sha256': old_sha,
                           'input_sha256': digest(old_input),
                           'basis': 'Complete task payload is unchanged (excluding bookkeeping hash/ID envelope); no new model execution is claimed.'}
        save_json(new_path, record)
        events.append({'unit_id': uid, 'role': role, 'target': reference(new_path), **record['reuse']})

    for role in ('solver', 'challenger'):
        copy_review(role, RUN/'reviews'/role/f'{old_key}.json', RUN/'reviews'/role/f'{key}.json',
                    blind_payload(old), blind_payload(new))
    ids = [q.id for q in new.questions]
    if ids != [q.id for q in old.questions]:
        raise ValueError('Cannot reuse reviews across changed question identity/order')
    for start in range(0, len(ids), 5):
        batch, n = ids[start:start+5], start//5+1
        copy_review('teaching', RUN/'reviews/teaching'/f'{old_key}-{n}.json',
                    RUN/'reviews/teaching'/f'{key}-{n}.json',
                    explanation_payload(old, batch), explanation_payload(new, batch))
    for q in new.questions:
        if q.kind != 'cross_document':
            continue
        for visible in document_subsets(new):
            a, b = ablation_payload(old, q.id, visible), ablation_payload(new, q.id, visible)
            path = RUN/'reviews/probe'/f'{"-".join(visible) or "none"}-{q.id}.json'
            if a != b or not path.exists():
                continue
            records = read(path)
            for record in records:
                if record['unit_id'] != uid or record['question_id'] != q.id or record['content_sha256'] != old_sha:
                    continue
                if record['input_sha256'] != digest(a):
                    raise ValueError('Original probe input does not match its recorded hash')
                validate_record(record, uid, old_sha, 'probe', RUN)
                validate_reuse(record, old, 'probe', RUN, question_id=q.id, visible=visible)
                history_path = RUN/'reviews/probe-history'/f'{old_key}-{path.name}'
                save_json(history_path, record)
                record['content_sha256'] = sha
                record['reuse'] = {'source': reference(history_path),
                                   'source_content_sha256': old_sha, 'input_sha256': digest(a),
                                   'basis': 'Entire one-question visible payload is unchanged; no new model execution.'}
                events.append({'unit_id': uid, 'role': 'probe', 'target': reference(path), **record['reuse']})
            save_json(path, records)
    log = RUN/'audit/reused-reviews.json'
    save_json(log, (read(log) if log.exists() else []) + events)
    print(f'{uid}: reused {len(events)} identical-input reviews; changed inputs remain pending')


def collect(uid, run_dir=None):
    RUN = Path(run_dir) if run_dir is not None else globals()["RUN"]
    spec = selected(RUN)[uid]
    run = read(RUN / 'run.json')
    freeze_instructions(RUN)
    key, sha = spec['key'], spec['content_sha256']
    author_model, author_effort = native_identity(RUN, 'author')
    review_model, review_effort = native_identity(RUN)
    unit = selected_unit(spec)
    results, actors, reuses = {}, {}, {}
    for role in ('solver', 'challenger', 'critic'):
        record = checked_record(RUN / 'reviews' / role / f'{key}.json', uid, sha, role, RUN)
        validate_reuse(record, unit, role, RUN)
        results[role], actors[role] = record['result'], record['agent']
        if 'reuse' in record:
            reuses[role] = record['reuse']
    if len(set(actors.values())) != 3:
        raise ValueError('Solver, challenger and critic must be different native agents')
    authors = [a['agent'] for a in read(RUN/'audit'/'agent-invocations.json')['agents']
               if a['role'] == 'author' and uid in a.get('assigned_units', [])
               and (a['model'], a['reasoning_effort']) == (author_model, author_effort)]
    if len(authors) != 1 or authors[0] in actors.values():
        raise ValueError('A unique recorded author and separate reviewers are required')
    teaching = []
    qids = [q.id for q in unit.questions]
    for start in range(0, len(qids), 5):
        batch = qids[start:start+5]
        record = checked_record(RUN / 'reviews' / 'teaching' / f'{key}-{start//5+1}.json', uid, sha, 'teaching', RUN)
        if record['agent'] in list(actors.values()) + authors:
            raise ValueError('Teaching review must use a separate native agent')
        validate_reuse(record, unit, 'teaching', RUN, question_ids=batch)
        payload = explanation_payload(unit, batch)
        teaching.append({'question_ids': batch, 'input_sha256': digest(payload),
                         'request_input_sha256': digest(payload),
                         'request_label': f'native-{record["agent"]}-teaching-{uid}-{start//5+1}',
                         'agent': record['agent'], 'result': record['result'],
                         **({'reuse': record['reuse']} if 'reuse' in record else {})})
    ablations = []
    for q in unit.questions:
        if q.kind != 'cross_document':
            continue
        for visible in document_subsets(unit):
            mask = '-'.join(visible) or 'none'
            records = read(RUN / 'reviews' / 'probe' / f'{mask}-{q.id}.json')
            matches = [r for r in records if r['unit_id'] == uid and r['question_id'] == q.id]
            if len(matches) != 1:
                raise ValueError(f'{uid}/{q.id}/{mask}: missing/duplicate probe')
            record = matches[0]
            validate_record(record, uid, sha, 'probe', RUN)
            validate_reuse(record, unit, 'probe', RUN, question_id=q.id, visible=visible)
            actors_log = read(RUN/'audit'/'agent-invocations.json')['agents']
            if not any(a['agent'] == record.get('agent') and a['role'] == 'probe'
                       and uid in a.get('assigned_units', [])
                       and (a['model'], a['reasoning_effort']) == (review_model, review_effort) for a in actors_log):
                raise ValueError('Probe agent not present in actual invocation ledger')
            if (record['content_sha256'], record['model'], record['reasoning_effort'], record['visible_document_ids']) != (sha, review_model, review_effort, visible):
                raise ValueError('Probe identity mismatch')
            payload = ablation_payload(unit, q.id, visible)
            if record['input_sha256'] != digest(payload):
                raise ValueError('Probe input mismatch')
            ablations.append({'question_id': q.id, 'visible_document_ids': visible,
                              'input_sha256': digest(payload), 'request_input_sha256': digest(payload),
                              'request_label': f'native-{record["agent"]}-probe-{uid}-{q.id}-{mask}',
                              'agent': record['agent'], 'result': record['result'],
                              **({'reuse': record['reuse']} if 'reuse' in record else {})})
    probe_agents = [r['agent'] for r in ablations]
    if len(set(probe_agents)) != len(probe_agents):
        raise ValueError('Each question/document-subset pair needs a distinct native agent context')
    exposed = set(actors.values()) | set(authors) | {r['agent'] for r in teaching}
    if exposed.intersection(probe_agents):
        raise ValueError('Document-removal agents must not be exposed to full content or other roles')
    provenance = {'model': author_model, 'effort': author_effort, 'execution': 'codex_builtin_subagent',
                  'quality_protocol': PROTOCOL, 'prompt_version': prompt_version(),
                  'accepted_at': datetime.now(timezone.utc).isoformat(),
                  'reviewer': {'model': review_model, 'effort': review_effort, 'execution': 'codex_builtin_subagent'},
                  'native_author_agent': authors[0], 'native_review_agents': actors, 'source_artifact': spec['canonical'],
                  'identical_input_review_reuse': reuses,
                  'student_input_sha256': digest(blind_payload(unit))}
    entry = {'checkpoint_version': 2, 'id': uid, 'blueprint': read(RUN/'plans'/f'{uid}.json'),
             'content': unit.model_dump(), 'content_sha256': sha,
             'quality': {'status': 'accepted_automated', 'shape_and_evidence': 'pass',
                         'blind_solution': results['solver'], 'option_challenge': results['challenger'],
                         'editorial_review': results['critic'], 'external_review': None,
                         'independent_audit': {'protocol': PROTOCOL, 'content_sha256': sha,
                                               'document_ablation': ablations, 'teaching_review': teaching}},
             'provenance': provenance}
    if errors := verify_entry(entry):
        raise ValueError(errors)
    history = History(RUN / 'novelty.sqlite3')
    history_id = f'{run["run_id"]}/{uid}/{sha}'
    prior = RUN / 'units' / f'{uid}.json'
    revises = read(prior).get('provenance', {}).get('novelty_id') if prior.exists() else None
    try:
        if not history.has_accepted(history_id, sha):
            if findings := history.check(unit, exclude_source=revises):
                raise ValueError({'novelty': findings})
            history.accept(history_id, unit, entry['blueprint'], revises=revises)
    finally:
        history.close()
    entry['quality']['novelty'] = 'pass'
    entry['provenance']['novelty_id'] = history_id
    save_json(RUN/'units'/f'{uid}.json', entry)
    print(f'{uid}: all native {review_model} review gates pass ({len(unit.questions)} questions)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN,
                        help='Native run directory (historical run is the default)')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('init')
    p.add_argument('--plans', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--model', choices=NATIVE_MODELS, default=MODEL)
    p.add_argument('--effort', choices=NATIVE_EFFORTS, default=EFFORT)
    p.add_argument('--external-effort', choices=EXTERNAL_EFFORTS, default=EXTERNAL_EFFORT)
    p = commands.add_parser('prepare'); p.add_argument('unit'); p.add_argument('source', type=Path)
    commands.add_parser('schemas'); commands.add_parser('probes')
    p = commands.add_parser('assignment'); p.add_argument('name'); p.add_argument('role'); p.add_argument('units', nargs='+')
    p = commands.add_parser('check-probe'); p.add_argument('name')
    p = commands.add_parser('probe-assignment'); p.add_argument('name'); p.add_argument('packet_name')
    p = commands.add_parser('reuse'); p.add_argument('unit'); p.add_argument('old_key')
    p = commands.add_parser('collect'); p.add_argument('units', nargs='+')
    args = parser.parse_args()
    if args.command == 'init': initialize(args.plans, args.run_id, args.seed, args.protocol, args.run_dir,
                                          model=args.model, effort=args.effort, external_effort=args.external_effort)
    elif args.command == 'prepare': prepare(args.unit, args.source, args.run_dir)
    elif args.command == 'schemas': schemas(args.run_dir)
    elif args.command == 'probes': probes(args.run_dir)
    elif args.command == 'assignment': assignment(args.name, args.role, args.units, args.run_dir)
    elif args.command == 'probe-assignment': probe_assignment(args.name, args.packet_name, args.run_dir)
    elif args.command == 'check-probe': check_probe(args.name, args.run_dir)
    elif args.command == 'reuse': reuse(args.unit, args.old_key, args.run_dir)
    else:
        for uid in args.units: collect(uid, args.run_dir)


if __name__ == '__main__':
    main()
