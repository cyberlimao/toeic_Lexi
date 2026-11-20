"""Native Sol author file bridge with fail-closed Claude API review.

This runner cannot call an author model. Student-material revisions return to
the native author; the only permitted model edit is a locked explanation edit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from toeic_engine.anthropic_provider import AnthropicProvider, ANTHROPIC_MODEL, EFFORT
from toeic_engine.engine import Engine, QualityError, content_hash, prompt, prompt_version, save_json, verify_entry
from toeic_engine.independent_audits import EvidenceDecision, TeachingReview, digest
from toeic_engine.models import BlindSolution, Challenge, ExplanationRevision, Review, Unit, blind_payload, validate_unit
from toeic_engine.novelty import History
from toeic_engine.provider import ProviderError

NATIVE = {'model': 'gpt-6.1-sol', 'effort': 'xhigh', 'execution': 'codex_builtin_subagent'}
CLAUDE = {'model': ANTHROPIC_MODEL, 'effort': EFFORT, 'thinking': 'adaptive', 'execution': 'anthropic_api'}
SCHEMAS = {'solver': BlindSolution, 'challenger': Challenge, 'critic': Review,
           'teaching_reviewer': TeachingReview, 'evidence_probe': EvidenceDecision,
           'explanation_editor': ExplanationRevision}


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        save_json(path, value)


def author_record(run, uid, agent):
    ledger = read(run / 'audit/agent-invocations.json')
    matches = [a for a in ledger['agents'] if a['agent'] == agent and a['role'] == 'author'
               and uid in a.get('assigned_units', []) and a['model'] == NATIVE['model']
               and a.get('reasoning_effort', a.get('effort')) == 'xhigh']
    if not matches:
        raise ValueError(f'{agent} is not an assigned native Sol/xhigh author for {uid}')
    return {**NATIVE, 'native_author_agent': agent}


def prepare(run: Path, uid: str, draft: Path, agent: str):
    run, draft = run.resolve(), draft.resolve()
    config = read(run / 'run.json')
    if config['prompt_version'] != prompt_version():
        raise ValueError('Current role prompts differ from the frozen run version')
    index = int(uid.removeprefix('u')) - 1
    if uid != f'u{index+1:03d}' or not 0 <= index < len(config['plans']):
        raise ValueError('Invalid unit ID')
    plan = config['plans'][index]
    unit = Unit.model_validate(read(draft))
    if errors := validate_unit(unit, plan):
        raise ValueError(errors)
    provenance = author_record(run, uid, agent)
    sha = content_hash(unit)
    key = f'{uid}-{sha[:12]}'
    canonical = run / 'drafts' / f'{key}.json'
    immutable_json(canonical, unit.model_dump())
    start = 101 + sum(p['question_count'] for p in config['plans'][:index])
    common = {'unit_id': uid, 'content_sha256': sha,
              'display_numbers': {q.id: start+i for i, q in enumerate(unit.questions)}}
    immutable_json(run / 'packets' / f'{key}.blind.json', {**common, 'student_material': blind_payload(unit)})
    immutable_json(run / 'packets' / f'{key}.content.json', {**common, 'content': unit.model_dump(), 'blueprint': plan})
    spec = {'key': key, 'content_sha256': sha, 'canonical': str(canonical),
            'source': str(draft), 'native_author': provenance,
            'student_input_sha256': digest(blind_payload(unit))}
    # An explanation edit carries its API editor record into later acceptance.
    sidecar = draft.with_suffix('.provenance.json')
    if sidecar.exists():
        edit = read(sidecar)
        if edit.get('content_sha256') != sha or edit.get('student_input_sha256') != spec['student_input_sha256']:
            raise ValueError('Explanation-editor provenance does not match draft')
        spec['explanation_edit'] = edit
        immutable_json(canonical.with_suffix('.provenance.json'), edit)
    with lock(run / '.locks/selected.lock'):
        state = read(run / 'selected.json') if (run / 'selected.json').exists() else {}
        state[uid] = spec
        save_json(run / 'selected.json', state)
    return unit, plan, spec


class RestrictedClaude:
    """Only exact repository review prompts/schemas; no full-unit author call.

    Validated results are reused only for an identical model, effort, prompt,
    payload, schema and actual output ceiling. Each event explicitly records
    whether an API call or an exact-input replay occurred.
    """
    model = NATIVE['model']
    author_provenance = NATIVE
    review_provenance = CLAUDE

    def __init__(self, run, uid, transport=None, max_calls=80, timeout=900):
        self.run, self.uid = Path(run).resolve(), uid
        self.api = self.run / 'api' / uid
        self.cache = self.run / 'request-cache' / uid
        self.events = self.run / 'review-events' / uid
        for folder in (self.api, self.cache, self.events):
            folder.mkdir(parents=True, exist_ok=True)
        self._transport = transport
        self.max_calls, self.timeout = max_calls, timeout
        self.allowed = {prompt(role): (role, schema) for role, schema in SCHEMAS.items()}

    def close(self):
        if self._transport is not None:
            self._transport.close()

    def generate(self, system, payload, schema, label, max_tokens=65536):
        allowed = self.allowed.get(system)
        if (not allowed or not isinstance(schema, type) or not issubclass(schema, allowed[1])
                or set(schema.model_fields) != set(allowed[1].model_fields) or schema is Unit):
            raise ProviderError('Native-author boundary: Claude cannot generate or replace a question unit')
        role = allowed[0]
        identity = {'provider': 'anthropic', **CLAUDE, 'system': system, 'payload': payload,
                    'schema': schema.model_json_schema(), 'max_tokens': max(65536, max_tokens)}
        request_hash = digest(identity)
        cache = self.cache / f'{request_hash}.json'
        event = {'role': role, 'label': label, 'request_sha256': request_hash,
                 'input_sha256': digest(payload), 'recorded_at': datetime.now(timezone.utc).isoformat()}
        with lock(self.cache / f'{request_hash}.lock'):
            if cache.exists():
                record = read(cache)
                if record['identity'] != identity or digest(record['result']) != record['result_sha256']:
                    raise ValueError('Cached review identity or content mismatch')
                result = schema.model_validate(record['result'])
                event.update(kind='identical_request_reuse', source=str(cache))
            else:
                if self._transport is None:
                    self._transport = AnthropicProvider(self.api, self.max_calls, self.timeout)
                    # A resumed process must append billing records, not erase
                    # the prior process's completed/failed API call history.
                    if (self.api / 'usage.json').exists():
                        self._transport.usage = read(self.api / 'usage.json')
                api_label = f'{role}-{request_hash}'
                result = self._transport.generate(system, payload, schema, api_label, max_tokens=max_tokens)
                result = schema.model_validate(result.model_dump())
                record = {'identity': identity, 'result': result.model_dump(),
                          'result_sha256': digest(result.model_dump()), 'api_label': api_label,
                          'api_request': str(self.api / f'{api_label}.request.json'),
                          'api_response': str(self.api / f'{api_label}.response.txt')}
                immutable_json(cache, record)
                event.update(kind='api_call', source=str(cache), api_label=api_label)
        event_id = digest(event)
        immutable_json(self.events / f'{event_id}.json', event)
        return result


def _publish_accepted(run, uid, entry, spec, execution):
    if errors := verify_entry(entry):
        raise ValueError(errors)
    if entry['content_sha256'] != spec['content_sha256']:
        raise ValueError('Accepted content differs from the selected native draft')
    entry['provenance'].update(spec['native_author'])
    entry['provenance'].update(source_artifact=spec['canonical'],
                               reviewer=CLAUDE, native_student_input_sha256=spec['student_input_sha256'],
                               novelty_id=f'{execution.resolve()}/{uid}', review_execution=str(execution.resolve()))
    if 'explanation_edit' in spec:
        entry['provenance']['explanation_editor'] = CLAUDE
        entry['provenance']['explanation_edit'] = spec['explanation_edit']
    save_json(run / 'units' / f'{uid}.json', entry)
    return entry


def review_unit(run, uid, draft, agent, *, transport=None, max_calls=80, timeout=900):
    run = Path(run).resolve()
    with lock(run / '.locks' / f'{uid}.lock'):
        unit, plan, spec = prepare(run, uid, Path(draft), agent)
        root = run / 'review-runs' / spec['key']
        root.mkdir(parents=True, exist_ok=True)
        immutable_json(root / 'input.json', {'content_sha256': spec['content_sha256'],
                                           'blueprint': plan, 'native_author': spec['native_author']})
        history_path = run / 'novelty.sqlite3'
        if not history_path.exists():
            raise ValueError('Initialize the run novelty database from the canonical history before review')
        history = History(history_path)
        provider = RestrictedClaude(run, uid, transport, max_calls, timeout)
        try:
            # Recover accepted work (including the durable pending checkpoint).
            executions = sorted(root.glob('execution-*'))
            for execution in executions:
                for checkpoint in [execution / 'units' / f'{uid}.json', execution / 'units' / f'{uid}.pending.json']:
                    if not checkpoint.exists():
                        continue
                    entry = read(checkpoint)
                    if errors := verify_entry(entry):
                        raise ValueError(errors)
                    if entry['content_sha256'] != spec['content_sha256']:
                        raise ValueError('Recovery checkpoint content mismatch')
                    hid = f'{execution.resolve()}/{uid}'
                    if not history.has_accepted(hid, spec['content_sha256']):
                        history.accept(hid, unit, plan, revises=entry['provenance'].get('revises'))
                    if '.pending.' in checkpoint.name:
                        checkpoint.replace(execution / 'units' / f'{uid}.json')
                    return _publish_accepted(run, uid, entry, spec, execution)
            rejected = list(root.glob('execution-*/attempts/*.rejection.json'))
            if rejected:
                raise QualityError(f'Unchanged draft already rejected; revise it rather than resampling the same review: {rejected[-1]}')
            prior_path = run / 'units' / f'{uid}.json'
            parent = read(prior_path)['provenance']['novelty_id'] if prior_path.exists() else None
            execution = root / f'execution-{len(executions)+1:03d}'
            execution.mkdir()
            save_json(execution / 'input.json', {'content_sha256': spec['content_sha256'], 'revises': parent})
            engine = Engine(provider, history, execution, max_attempts=1)
            entry = engine.generate_unit(plan, uid, initial=unit, revises=parent,
                                         initial_author_provenance=spec['native_author'])
            return _publish_accepted(run, uid, entry, spec, execution)
        finally:
            provider.close()
            history.close()


def edit_explanations(run, uid, draft, agent, feedback, question_ids, out, *, transport=None, max_calls=10, timeout=900):
    run, out = Path(run).resolve(), Path(out).resolve()
    with lock(run / '.locks' / f'{uid}.lock'):
        unit, plan, spec = prepare(run, uid, Path(draft), agent)
        if out.exists() or out.with_suffix('.provenance.json').exists():
            raise ValueError('Use a new output path for each explanation revision')
        if not question_ids or len(set(question_ids)) != len(question_ids) or not set(question_ids) <= {q.id for q in unit.questions}:
            raise ValueError('Invalid explanation repair IDs')
        feedback_text = Path(feedback).read_text()
        if not feedback_text.strip():
            raise ValueError('Explanation repair requires concrete feedback')
        payload = {'blueprint': plan, 'rejected_draft': unit.model_dump(),
                   'required_fixes': [feedback_text], 'repair_question_ids': question_ids}
        provider = RestrictedClaude(run, uid, transport, max_calls, timeout)
        try:
            replacement = provider.generate(prompt('explanation_editor'), payload, ExplanationRevision,
                                            f'{spec["key"]}-explanation-edit')
        finally:
            provider.close()
        if [r.question_id for r in replacement.replacements] != question_ids:
            raise ValueError('Editor did not return the exact requested IDs in order')
        new = unit.model_copy(deep=True)
        changes = {r.question_id: r.explanation for r in replacement.replacements}
        for q in new.questions:
            if q.id in changes:
                q.explanation = changes[q.id]
        assert blind_payload(new) == blind_payload(unit)
        assert [q.answer for q in new.questions] == [q.answer for q in unit.questions]
        if errors := validate_unit(new, plan):
            raise ValueError(errors)
        record = {'editor': CLAUDE, 'native_author': spec['native_author'], 'unit_id': uid,
                  'base_content_sha256': spec['content_sha256'], 'content_sha256': content_hash(new),
                  'student_input_sha256': spec['student_input_sha256'], 'question_ids': question_ids,
                  'feedback': str(Path(feedback).resolve()), 'student_material_and_keys_unchanged': True,
                  'status': 'requires_complete_review'}
        if 'explanation_edit' in spec:
            record['prior_explanation_edit'] = spec['explanation_edit']
        immutable_json(out, new.model_dump())
        immutable_json(out.with_suffix('.provenance.json'), record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'review-unit', 'edit-explanations'):
        p = sub.add_parser(command)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--unit', required=True)
        p.add_argument('--draft', type=Path, required=True)
        p.add_argument('--author-agent', required=True)
        if command != 'prepare':
            p.add_argument('--max-calls', type=int, default=80 if command == 'review-unit' else 10)
            p.add_argument('--timeout', type=int, default=900)
        if command == 'edit-explanations':
            p.add_argument('--feedback', type=Path, required=True)
            p.add_argument('--questions', required=True)
            p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        common = (args.run, args.unit, args.draft, args.author_agent)
        if args.command == 'prepare':
            _, _, result = prepare(*common)
        elif args.command == 'review-unit':
            entry = review_unit(*common, max_calls=args.max_calls, timeout=args.timeout)
            result = {'unit_id': args.unit, 'status': 'accepted_automated', 'content_sha256': entry['content_sha256']}
        else:
            result = edit_explanations(*common, args.feedback, args.questions.split(','), args.out,
                                       max_calls=args.max_calls, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False))
    except (QualityError, ProviderError, ValueError) as exc:
        print(f'{type(exc).__name__}: {exc}')
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
