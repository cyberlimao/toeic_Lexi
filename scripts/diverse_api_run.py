"""Claude API edition planning and hash-bound diversity certification.

``init`` is offline. Generation uses complete_planned_run and its unchanged
Engine gates. The additional annotate/whole-review commands make fresh Claude
API requests; an API response and usage record are required for certification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import uuid

from pydantic import Field

from toeic_engine.anthropic_provider import AnthropicProvider, ANTHROPIC_MODEL, EFFORT, anthropic_schema
from toeic_engine.diversity import DiversityHistory, FingerprintAnnotation, content_hash, literal, validate_annotations
from toeic_engine.engine import PROMPT_NAMES, prompt, prompt_version, save_json, verify_entry
from toeic_engine.independent_audits import digest
from toeic_engine.models import Model, Unit
from toeic_engine.novelty import History, unit_texts
from toeic_engine.planning import Planner
from scripts.diverse_native_run import AVOIDANCE, whole_test_metrics, save_accepted_history

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'api-reviewed-fingerprints-and-whole-test-v1'
LEGACY_INPUT_VERSION = 'api-editorial-input-v1'
INPUT_VERSION = 'api-editorial-input-v2'
INPUT_CONFIG_PATH = Path('audit/api-editorial-input-v2.json')
ADJUDICATION_VERSION = 'fingerprint-q25-editorial-adjudication-v1'
WHOLE_ADJUDICATION_VERSION = 'whole-review-fingerprint-adjudication-context-v1'

FINGERPRINT_PROMPT = '''You are an independent Claude Opus 5.5/xhigh editorial fingerprint reviewer. You did not author this unit and receive no solver/challenger/critic/Astra conclusions. Inspect every actual question and all options against the supplied frozen blueprint and historical examples. Return one precise fingerprint annotation per question in order. Classify the operation really required to select the answer, not the operation claimed by its label or explanation. A sufficient two-document evidence path is not proof of necessary dependence. Explicit attitude paraphrase or combining two stated duties is not implicit inference. Give specific stable microconstruct/option-family/solution-template codes; do not invent synonyms merely to evade historical matches. answer_lexeme is the literal normalized keyed option: lowercase English words with internal apostrophes/hyphens preserved. Part 5 has no passage evidence or evidence_join; passage questions need exact quoted evidence and an evidence_join document_count matching the distinct cited documents. Set each reviewer_id to the supplied reviewer_id and give a concrete reviewer_note of at least 20 characters. Report content/construct mismatches in issues and verdict revise; do not silently change the question or relabel it to satisfy its blueprint. Common core skills may recur, but wording alone does not establish a new reasoning template. This is a content review, not psychometric calibration. Return only the requested JSON schema.'''

WHOLE_PROMPT = '''You are the independent Claude Opus 5.5/xhigh final whole-test editor. Inspect the supplied 100-question reading test, frozen plans, reviewed fingerprints and quantitative diversity findings as a coherent test. Check actual microconstruct realization, natural and varied document purposes/voices, question sequence, distinct reasoning operations, cross-edition reuse and answer-option specificity/length clues. Do not accept label counts as evidence. Distinguish explicit detail from inference and available evidence paths from indispensable cross-document joins; consider ordinary test-taking elimination and common English role meanings. A correct keyed answer does not prove a sound question or a correct distractor explanation. Never pad choices just to lower a length statistic. Difficulty is an editorial estimate, not calibrated by examinees. Do not alter questions or explanations. Copy every supplied content binding exactly. Pass only when all substantive defects are resolved. Every quantitative editorial_alert needs a specific resolved_alerts entry citing affected question numbers and why it is acceptable. Assess each requires_review repetition finding against its prior example: only justified core-skill recurrence may receive a core_skill_reuse entry with exact unit/question/finding ID and a concrete comparative reason of at least 30 characters. Lexical-copy findings cannot be waived. Do not claim heuristically unknown archive semantics are known. Return revise for actual remaining issues, including materially unmet planned constructs. No external/Astra judgments are supplied. Return only the requested JSON schema.'''


class FingerprintReview(Model):
    verdict: Literal['pass', 'revise']
    issues: list[str]
    annotations: list[FingerprintAnnotation]


class ContentBinding(Model):
    unit_id: str
    content_sha256: str


class AlertDisposition(Model):
    alert: str
    reason: str = Field(min_length=30)


class CoreSkillReuse(Model):
    unit_id: str
    question_id: str
    finding_id: str
    reason: str = Field(min_length=30)


class WholeFinding(Model):
    question_numbers: list[int]
    category: str
    decision: Literal['pass', 'revise', 'editorial_note']
    explanation: str = Field(min_length=30)


class WholeReview(Model):
    verdict: Literal['pass', 'revise']
    content_bindings: list[ContentBinding]
    unresolved_issues: list[str]
    resolved_alerts: list[AlertDisposition]
    core_skill_reuse: list[CoreSkillReuse]
    findings: list[WholeFinding] = Field(min_length=1)
    limitations: str = Field(min_length=30)


class AnnotationValidationError(ValueError):
    """Only post-response annotation shape/quotation checks; never provenance."""
    def __init__(self, unit_id, errors):
        self.unit_id, self.errors = unit_id, errors
        super().__init__(f'{unit_id}: invalid reviewed annotations: {errors}')


def read(path):
    return json.loads(Path(path).read_text())


def initialize(run: Path, seed=None, corpus: Path | None = None, *, run_id=None, examples: Path = ROOT/'examples'):
    run = run.resolve()
    corpus = corpus.resolve() if corpus is not None else None
    if run.exists() and any(run.iterdir()):
        raise ValueError('Use a new empty run directory; initialization never replaces an existing run')
    run.mkdir(parents=True, exist_ok=True)
    semantic = DiversityHistory(run/'audit/history-before.json')
    archives = semantic.bootstrap_archives(examples)
    lexical = History(run/'novelty.sqlite3')
    try:
        imported = lexical.import_corpus(corpus) if corpus is not None else 0
        # Import prior releases as historical material, not newly accepted
        # candidates. Their historical overlap must not invalidate the import.
        with lexical.db:
            for source in archives['archives']:
                data = read(source['path'])
                for entry in data['units']:
                    unit = Unit.model_validate(entry['content'])
                    identity = f"archive:{data['run_id']}:{entry['id']}"
                    for local_id, text in unit_texts(unit):
                        lexical._add(f'{identity}:{local_id}', text)
                    lexical.db.execute('INSERT INTO accepted(id,summary,plan,content_sha256) VALUES (?,?,?,?)',
                                       (identity, unit.scenario_summary, json.dumps(entry['blueprint']), content_hash(unit)))
        usage = lexical.usage()
        usage.update(semantic.usage())
    finally:
        lexical.close()
    planner = Planner(seed, usage)
    plans = planner.build('full')
    for plan in plans:
        plan['cross_edition_editorial_context'] = {
            'avoidance': AVOIDANCE, 'observed_history': semantic.planning_context(plan['part'], 16),
            'limitations': archives['limitations'],
            'instruction': 'Historical snippets are avoidance evidence, never source text to imitate. Realize the planned microconstruct and vary the actual solution operation, not only names or setting.'}
    instructions = {name: prompt(name) for name in PROMPT_NAMES}
    for name, text in instructions.items():
        target = run/'instructions'/f'{name}.txt'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    save_json(run/'audit/archive-sources.json', archives)
    save_json(run/'audit/benchmark-novelty-sources.json', {
        'imported_texts': imported, 'sources': [
            {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(corpus.glob('test-*/test-*.json'))] if corpus is not None else []})
    save_json(run/'audit/editorial-avoidance.json', AVOIDANCE)
    save_json(run/'audit/planned-test.json', plans)
    save_json(run/'unit.schema.json', Unit.model_json_schema())
    save_json(run/'run.json', {
        'run_id': run_id or run.name, 'created_at': datetime.now(timezone.utc).isoformat(),
        'seed': planner.seed, 'part': 'full', 'plans': plans,
        'prompt_version': prompt_version(), 'history': str(run/'novelty.sqlite3'),
        'corpus': str(corpus) if corpus is not None else None,
        'author_provider': 'claude', 'review_provider': 'claude',
        'author_model': ANTHROPIC_MODEL, 'review_model': ANTHROPIC_MODEL, 'reasoning_effort': EFFORT,
        'execution': 'anthropic_api', 'diversity_protocol': PROTOCOL,
        'frozen_history_sha256': hashlib.sha256((run/'audit/history-before.json').read_bytes()).hexdigest(),
    })
    save_json(run/'prompt-snapshot.json', {'prompt_version': prompt_version(), 'prompts': instructions,
        'model': ANTHROPIC_MODEL, 'effort': EFFORT, 'thinking': 'adaptive', 'execution': 'anthropic_api'})
    print(f'Offline Claude API plan ready: {run}; {len(plans)} units / 100 questions; seed {planner.seed}', flush=True)


def configuration(run):
    config = read(run/'run.json')
    if (config.get('diversity_protocol'), config.get('author_provider'), config.get('review_provider'),
            config.get('author_model'), config.get('review_model'), config.get('reasoning_effort')) != (
            PROTOCOL, 'claude', 'claude', ANTHROPIC_MODEL, ANTHROPIC_MODEL, EFFORT):
        raise ValueError('Not a supported all-Claude API diversity run')
    if config['frozen_history_sha256'] != hashlib.sha256((run/'audit/history-before.json').read_bytes()).hexdigest():
        raise ValueError('Frozen history changed')
    if config['plans'] != read(run/'audit/planned-test.json'):
        raise ValueError('Run blueprints differ from the frozen planned test; an explicit reviewed plan-amendment mechanism is required')
    if config['prompt_version'] != prompt_version():
        raise ValueError('Role prompts changed after run initialization')
    return config


def freeze_supplemental(run):
    """The API protocol has its own prompts, never native-agent instructions."""
    expected = {'fingerprint_reviewer': FINGERPRINT_PROMPT, 'whole_test_reviewer': WHOLE_PROMPT}
    path = run/'audit/api-editorial-prompts.json'
    record = {'protocol': PROTOCOL, 'prompts': expected, 'sha256': digest(expected)}
    if path.exists():
        if read(path) != record:
            raise ValueError('API editorial instructions changed after freezing')
    else:
        save_json(path, record)
    return record


def input_configuration_record():
    """Explicit interpretation of an ambiguous planner field, not a plan edit.

    The observed ItemReview boolean retains its factual meaning. The planner's
    field allocates a minimum requirement and does not forbid additional
    cross-sentence work. Exact source snapshots make that interpretation
    inspectable without rewriting any earlier API response or frozen prompt.
    """
    sources = [
        ('toeic_engine/planning.py', '_part6_slot',
         '"requires_cross_sentence_context": skill in ("contextual_vocabulary", "sentence_cohesion")'),
        ('toeic_engine/planning.py', 'P6_VARIANTS.reference_cohesion.event_reference',
         'Refer back to an established action or event; do not use a generic this process as the only context claim.'),
        ('toeic_engine/planning.py', 'P6_VARIANTS.reference_cohesion.entity_chain',
         'Track a person or object across sentences without repeating its name beside the blank.'),
        ('toeic_engine/models.py', 'validate_review',
         'if unit.part == 6 and sum(r.requires_cross_sentence_context for r in review.item_reviews) < 2:'),
        ('toeic_engine/prompts/author.txt', 'Part 6',
         'At least two items should require surrounding discourse.'),
    ]
    evidence = []
    for path, symbol, quotation in sources:
        raw = (ROOT/path).read_bytes()
        if quotation not in raw.decode():
            raise ValueError(f'Blueprint interpretation source changed: {path}/{symbol}')
        evidence.append({'path':path, 'sha256':hashlib.sha256(raw).hexdigest(),
                         'symbol_or_section':symbol, 'exact_excerpt':quotation})
    return {
        'version':INPUT_VERSION,
        'kind':'explicit_blueprint_field_semantics_clarification',
        'reason':'The first u002 fingerprint review identified an ambiguity between a planner minimum-allocation flag and the actual dependency observed by the critic. Preserve that original revise judgment. This clarification applies prospectively to new API requests; it does not retroactively turn any review into pass.',
        'blueprint_flag_interpretation':{
            'field':'Part 6 blueprint.slots[].requires_cross_sentence_context',
            'true':'This slot is explicitly allocated to the minimum set that must require context beyond its own sentence.',
            'false':'This slot is not counted in that minimum allocation. It is not a prohibition on using neighboring sentences and does not require that the item be solvable locally.',
            'construct_precedence':'Realize the slot microconstruct, realization_strategy and reading_operation. A reference-cohesion slot requiring an established event or cross-sentence entity must retain that reference dependency; do not remove it merely to make a false planner flag look like a factual local-answerability claim.',
            'observed_review_field':'ItemReview.requires_cross_sentence_context is a separate factual judgment of the realized item. Record true when removing neighboring sentences removes a necessary referent/chronology cue or leaves multiple viable alternatives, regardless of the planner allocation flag. Do not copy the blueprint boolean into the review.',
            'limits':'This clarifies an ambiguous metadata field. It does not waive answer uniqueness, factual explanation accuracy, the at-least-two actual context-dependent items gate, diversity checks, or any required microconstruct. Report genuine conflicts with the planned construct normally.',
        },
        'source_evidence':evidence,
        'unchanged_artifacts':['run.json plans','audit/planned-test.json','audit/api-editorial-prompts.json','all prior API requests and responses','all prior annotation verdicts'],
    }


def freeze_input_configuration(run):
    expected = input_configuration_record()
    path = run/INPUT_CONFIG_PATH
    if path.exists():
        if read(path) != expected:
            raise ValueError('Versioned API input configuration changed')
    else:
        save_json(path, expected)
    return expected


def with_input_configuration(run, payload, version):
    if version == LEGACY_INPUT_VERSION:
        return payload
    if version != INPUT_VERSION:
        raise ValueError('Unsupported API editorial input version')
    path = run/INPUT_CONFIG_PATH
    if not path.exists() or read(path) != input_configuration_record():
        raise ValueError('Missing or changed versioned API input configuration')
    return {**payload, 'editorial_input_configuration':read(path)}


def selected_entries(run):
    config = configuration(run)
    selections_path = run/'audit/selected-checkpoints.json'
    selections = read(selections_path) if selections_path.exists() else {}
    if set(selections) - {f'u{i:03d}' for i in range(1, len(config['plans'])+1)}:
        raise ValueError('Unknown selected checkpoint unit')
    entries = []
    for i, plan in enumerate(config['plans'], 1):
        uid = f'u{i:03d}'
        path = Path(selections[uid]) if uid in selections else run/'units'/f'{uid}.json'
        if not path.is_absolute():
            path = ROOT/path
        if not path.exists():
            if uid in selections:
                raise ValueError(f'Missing selected checkpoint {uid}')
            continue
        entry = read(path)
        if errors := verify_entry(entry):
            raise ValueError(f'{uid}: {errors}')
        if entry['blueprint'] != plan:
            raise ValueError(f'{uid}: selected checkpoint blueprint differs from current declared plan')
        provenance = entry['provenance']
        if provenance.get('prompt_version') != config['prompt_version']:
            raise ValueError(f'{uid}: checkpoint uses a different frozen prompt version')
        if (provenance.get('model'), provenance.get('effort'), provenance.get('reviewer', {}).get('model'),
                provenance.get('reviewer', {}).get('effort')) != (ANTHROPIC_MODEL, EFFORT, ANTHROPIC_MODEL, EFFORT):
            raise ValueError(f'{uid}: mixed author/reviewer provenance')
        entry['id'] = uid
        entries.append(entry)
    return entries


def api_call(run, role, payload, schema, *, max_calls=3, timeout=900, transport=None):
    """Store a unique request/response/usage chain even for rejected judgments."""
    frozen = freeze_supplemental(run)
    system = frozen['prompts'][role]
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    folder = run/'api-editorial'/invocation
    provider = transport or AnthropicProvider(folder, max_calls, timeout)
    label = role
    try:
        result = provider.generate(system, payload, schema, label, max_tokens=65536)
    finally:
        provider.close()
    evidence = {'role': role, 'model': ANTHROPIC_MODEL, 'effort': EFFORT, 'execution': 'anthropic_api',
                'system_sha256': hashlib.sha256(system.encode()).hexdigest(), 'payload_sha256': digest(payload),
                'request': str((folder/f'{label}.request.json').relative_to(run)),
                'response': str((folder/f'{label}.response.txt').relative_to(run)),
                'usage': str((folder/'usage.json').relative_to(run)), 'label': label,
                'result': result.model_dump()}
    for field in ('request', 'response', 'usage'):
        evidence[field+'_sha256'] = hashlib.sha256((run/evidence[field]).read_bytes()).hexdigest()
    save_json(folder/'evidence.json', evidence)
    validate_api_evidence(run, evidence, role, payload, schema)
    return result, str((folder/'evidence.json').relative_to(run))


def validate_api_evidence(run, evidence, role, payload, schema):
    frozen = freeze_supplemental(run)
    system = frozen['prompts'][role]
    if (evidence.get('model'), evidence.get('effort'), evidence.get('execution'), evidence.get('role')) != (
            ANTHROPIC_MODEL, EFFORT, 'anthropic_api', role):
        raise ValueError('Missing actual Claude API editorial provenance')
    if evidence.get('payload_sha256') != digest(payload) or evidence.get('system_sha256') != hashlib.sha256(system.encode()).hexdigest():
        raise ValueError('API editorial input or prompt mismatch')
    for field in ('request', 'response', 'usage'):
        if hashlib.sha256((run/evidence[field]).read_bytes()).hexdigest() != evidence[field+'_sha256']:
            raise ValueError(f'API {field} evidence hash mismatch')
    request = read(run/evidence['request'])
    wire = request['wire_request']
    if (request.get('model'), request.get('effort'), request.get('system'), request.get('payload'), request.get('schema')) != (
            ANTHROPIC_MODEL, EFFORT, system, payload, anthropic_schema(schema)):
        raise ValueError('API request does not match independently reconstructed inputs')
    if (wire.get('model'), wire.get('thinking'), wire.get('output_config'), wire.get('system'), wire.get('messages')) != (
            ANTHROPIC_MODEL, {'type': 'adaptive'}, {'effort': EFFORT, 'format': {'type':'json_schema','schema':anthropic_schema(schema)}},
            system, [{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]):
        raise ValueError('Wire model/effort/schema/conversation mismatch')
    if wire.get('stream') is not True or wire.get('max_tokens', 0) < 65536:
        raise ValueError('Wire thinking/output allowance or streaming differs from the API protocol')
    response = (run/evidence['response']).read_text()
    parsed = schema.model_validate_json(response)
    if parsed.model_dump() != evidence.get('result'):
        raise ValueError('Stored editorial result differs from actual API response')
    good = [u for u in read(run/evidence['usage']) if u.get('label') == evidence['label']
            and u.get('model_version') == ANTHROPIC_MODEL and u.get('effort') == EFFORT
            and u.get('stream_complete') and u.get('finish_reason') == 'end_turn'
            and u.get('content_sha256') == hashlib.sha256(response.encode()).hexdigest()]
    if not good:
        raise ValueError('No completed API usage record binds this editorial result')
    return parsed


def fingerprint_payload(run, entry, *, input_version=INPUT_VERSION):
    uid, sha = entry['id'], entry['content_sha256']
    config = configuration(run)
    payload = {'unit_id': uid, 'content_sha256': sha,
            'reviewer_id': f'anthropic-api:{uid}:{sha[:12]}:fingerprint',
            'blueprint': config['plans'][int(uid[1:])-1], 'unit': entry['content'],
            'historical_examples': DiversityHistory(run/'audit/history-before.json').planning_context(entry['content']['part'], 24)}
    return with_input_configuration(run, payload, input_version)


def annotate(run, units=None, *, max_calls=3, timeout=900, repair_invalid=False):
    freeze_input_configuration(run)
    entries = selected_entries(run)
    if units and set(units) - {e['id'] for e in entries}:
        raise ValueError('Requested annotation unit has no valid selected checkpoint')
    for entry in entries:
        if units and entry['id'] not in units:
            continue
        uid, sha = entry['id'], entry['content_sha256']
        path = run/'annotations'/f'{uid}-{sha[:12]}.reviewed.json'
        if path.exists():
            try:
                load_annotation(run, entry)
            except AnnotationValidationError:
                if not repair_invalid:
                    raise
                repair_annotation_format(run, entry, max_calls=max_calls, timeout=timeout)
            print(f'{uid}: exact-version API fingerprint review retained', flush=True)
            continue
        payload = fingerprint_payload(run, entry)
        result, evidence = api_call(run, 'fingerprint_reviewer', payload, FingerprintReview, max_calls=max_calls, timeout=timeout)
        packet = {'unit_id':uid, 'content_sha256':sha, 'model':ANTHROPIC_MODEL, 'effort':EFFORT,
                  'execution':'anthropic_api', 'api_evidence':evidence,
                  'editorial_input_version':INPUT_VERSION, **result.model_dump()}
        save_json(path, packet)
        load_annotation(run, entry)
        print(f'{uid}: API fingerprint review {result.verdict}', flush=True)


def annotation_errors(entry, packet):
    errors = validate_annotations(Unit.model_validate(entry['content']), packet['annotations'])
    expected_actor = f"anthropic-api:{entry['id']}:{entry['content_sha256'][:12]}:fingerprint"
    for row in packet['annotations']:
        if row['reviewer_id'] != expected_actor:
            errors.append(f"{row['question_id']}: reviewer_id must equal the assigned API fingerprint reviewer identity")
    return errors


def format_correction(entry, previous, errors):
    """Permit narrow metadata/quotation repairs, never a new semantic verdict."""
    allowed = {}
    for error in errors:
        match = re.match(r'^(q\d+): (.*)$',error)
        if not match:
            raise ValueError(f'Annotation error is not an eligible narrow format repair: {error}')
        qid, detail = match.groups()
        if detail=='answer_lexeme must equal the literal normalized keyed option':
            field='answer_lexeme'
        elif detail=='evidence_join document_count must match cited documents':
            field='evidence_join.document_count'
        elif detail.startswith('Invalid evidence ') and 'quote is not an exact substring' in detail:
            field='evidence'
        elif detail=='reviewer_id must equal the assigned API fingerprint reviewer identity':
            field='reviewer_id'
        else:
            raise ValueError(f'Annotation error is not an eligible narrow format repair: {error}')
        allowed.setdefault(qid,set()).add(field)
    if not allowed:
        raise ValueError('No eligible annotation format errors; semantic revise judgments are not retried')
    keys = {q['id']:literal(next(o['text'] for o in q['options'] if o['label']==q['answer'])) for q in entry['content']['questions']}
    return {
        'protocol':'fingerprint-narrow-format-repair-v1',
        'validation_errors':errors,
        'allowed_fields':[{'question_id':qid,'fields':sorted(fields)} for qid,fields in sorted(allowed.items())],
        'required_literal_values':[{'question_id':qid,'answer_lexeme':keys[qid]} for qid,fields in sorted(allowed.items()) if 'answer_lexeme' in fields],
        'previous_response':{k:previous[k] for k in ('verdict','issues','annotations')},
        'instruction':'Repair only the listed annotation metadata or exact quotation fields and return the complete response. Keep verdict and issues exactly unchanged, including every substantive revise finding. Preserve all other annotation fields, question coverage and order. Do not reassess, clear or add semantic judgments; this is an explicit format correction to the prior response, not another sampling attempt for a better verdict. No student content may change.',
    }


def check_narrow_changes(previous, current, correction):
    if current['verdict']!=previous['verdict'] or current['issues']!=previous['issues']:
        raise ValueError('Format repair changed the original semantic verdict/issues')
    allowed = {r['question_id']:set(r['fields']) for r in correction['allowed_fields']}
    if [a['question_id'] for a in current['annotations']] != [a['question_id'] for a in previous['annotations']]:
        raise ValueError('Format repair changed annotation coverage/order')
    for before,after in zip(previous['annotations'],current['annotations']):
        baseline, candidate = copy.deepcopy(before),copy.deepcopy(after)
        for field in allowed.get(before['question_id'],set()):
            if field=='evidence_join.document_count':
                if not isinstance(candidate.get('evidence_join'),dict) or not isinstance(baseline.get('evidence_join'),dict):
                    raise ValueError('Format repair cannot replace an evidence-join judgment')
                candidate['evidence_join']['document_count']=baseline['evidence_join']['document_count']
            else:
                candidate[field]=baseline[field]
        if baseline!=candidate:
            raise ValueError(f"Format repair changed a non-permitted annotation field: {before['question_id']}")


def archived_packet(run, source, seen):
    path = (run/source['annotation_archive']).resolve()
    if not path.is_relative_to((run/'annotations/history').resolve()):
        raise ValueError('Annotation archive must be inside immutable history')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != source['annotation_sha256']:
        raise ValueError('Original invalid annotation history changed')
    seen = set() if seen is None else set(seen)
    if sha in seen or len(seen) > 8:
        raise ValueError('Invalid annotation repair history cycle')
    return read(path), seen | {sha}


def adjudication_paths(entry):
    key = f"{entry['id']}-{entry['content_sha256']}"
    return (Path('audit')/f'{key}.q25-adjudication-v1.json',
            Path('annotations/adjudications')/f'{key}.q25-reservation.json')


def adjudication_contract(run, entry):
    """Source-backed scope plus explicitly NEW editorial intent, never a pass."""
    config = configuration(run)
    if entry['id'] != 'u001' or entry['content']['part'] != 5 or len(entry['content']['questions']) != 30:
        raise ValueError('This adjudication is limited to full Part 5 u001/q25')
    plan = config['plans'][0]
    slot = plan['slots'][24]
    strategy = 'Use a stated restricted set to distinguish another, other, or the other naturally.'
    if (slot['id'], slot['microconstruct'], slot['realization_strategy']) != ('q25', 'determiner_quantifier.restricted_set', strategy) or 'option_set' in slot or 'reading_operation' in slot or slot.get('family')!='reference_quantity':
        raise ValueError('Adjudication does not match the original restricted-set slot')
    sources = []
    quotations = [
        ('toeic_engine/planning.py', 'P5_VARIANTS.determiner_quantifier.restricted_set', strategy),
        ('toeic_engine/prompts/author.txt', 'Whole-test diversity revision / passage groups',
         'Cross-document integration can identify a referent, connect a response to an earlier request, reconcile a correction, explain a consequence, or track a sequence; do not turn every group into a four-candidate eligibility matrix.'),
    ]
    for name, section, quote in quotations:
        raw = (ROOT/name).read_bytes()
        if quote not in raw.decode():
            raise ValueError('Adjudication source excerpt changed')
        sources.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),
                        'section':section,'exact_excerpt':quote})
    avoidance = plan['cross_edition_editorial_context']['avoidance']
    avoided = 'Four candidates each failing a custom eligibility constraint'
    if (avoided not in avoidance['part7'] or avoided in avoidance['part5']
            or 'editorial avoidance priorities for this edition, not permanent bans' not in avoidance['limitations']):
        raise ValueError('Adjudication Part 7 scope source changed')
    for path in (run/'run.json', run/'audit/planned-test.json'):
        sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'section':'plans[0] (or [0]).slots[24] and cross_edition_editorial_context.avoidance',
                        'exact_json_excerpt':{'slot':slot,'avoidance':avoidance}})
    return {'version':ADJUDICATION_VERSION,'unit_id':entry['id'],'content_sha256':entry['content_sha256'],
        'question_id':'q25','source_evidence':sources,
        'source_backed_scope':{
            'part7_avoidance':'The four-candidate/custom-eligibility pattern appears under avoidance.part7 and in the author paragraph about passage/cross-document groups. It is not an automatic prohibition on Part 5 restricted-set quantifier reference. The separately preserved limitations call these priorities, not permanent bans.',
            'literal_strategy':'The original natural-language strategy says another, other, OR the other. It declares no fixed option_set. The original source alone does not explicitly enumerate every permissible alternative option family.',
            'declared_operation':'This frozen slot has family reference_quantity and microconstruct determiner_quantifier.restricted_set, with no reading_operation field. It does not expressly require local_grammar or prohibit necessary contextual set/reference interpretation. Judge the actual operation against the declared construct, not an invented pure-form-only operation.'},
        'new_explicit_editorial_intent':{
            'status':'Prospective editorial-intent clarification in this adjudication; not a claim that the source already contained this exact rule.',
            'scope':'The other may be contrasted with other quantifiers rather than requiring another and other in the same option set. The stated restricted set and member reference must still be necessary to distinguish the uniquely keyed quantity/reference. A merely grammatical fit or an unrelated operational eligibility puzzle does not satisfy this construct.',
            'no_prejudged_outcome':'Independently assess whether this actual q25 realizes that reference contrast and has natural concise English, a coherent time frame, a unique answer, accurate explanations and credible distractors. The original secondary length/tense concern remains open for judgment. Real construct, naturalness or distractor defects still require revise. No student content, plan, prompt or threshold changes are authorized.'}}


def check_adjudicated_changes(previous, current):
    expected = [f'q{i}' for i in range(1,31)]
    if [a['question_id'] for a in previous['annotations']] != expected or [a['question_id'] for a in current['annotations']] != expected:
        raise ValueError('Adjudication changed annotation coverage/order')
    if any(a != b for a,b in zip(previous['annotations'],current['annotations']) if a['question_id'] != 'q25'):
        raise ValueError('Adjudication changed one of the locked other 29 annotations')
    if (current['verdict'] == 'pass') != (not current['issues']):
        raise ValueError('Adjudication verdict and issues disagree')
    if any(not re.match(r'^q25\b', issue) for issue in current['issues']):
        raise ValueError('Adjudication issues must be explicitly limited to q25')


def adjudication_context(run, entry, previous, source):
    contract_path, reservation_path = adjudication_paths(entry)
    if source.get('contract') != str(contract_path) or source.get('reservation') != str(reservation_path):
        raise ValueError('Unexpected adjudication contract/reservation location')
    for key in ('contract','reservation'):
        if hashlib.sha256((run/source[key]).read_bytes()).hexdigest() != source[key+'_sha256']:
            raise ValueError(f'Adjudication {key} hash mismatch')
    contract = adjudication_contract(run,entry)
    if read(run/contract_path) != contract:
        raise ValueError('Adjudication source contract/excerpts changed')
    reservation = {'protocol':ADJUDICATION_VERSION,'unit_id':entry['id'],'content_sha256':entry['content_sha256'],
                   'question_id':'q25','source':{k:source[k] for k in ('annotation_archive','annotation_sha256','contract','contract_sha256')}}
    if read(run/reservation_path) != reservation:
        raise ValueError('Adjudication reservation changed')
    if previous['verdict'] != 'revise' or not previous['issues'] or any(not re.match(r'^q25\b', i) for i in previous['issues']):
        raise ValueError('Original adjudication issues are not exclusively q25 revise findings')
    if annotation_errors(entry, previous):
        raise ValueError('Resolve original formatting before semantic adjudication')
    return {'protocol':ADJUDICATION_VERSION,'source':source,'contract':contract,
        'previous_response':{k:previous[k] for k in ('verdict','issues','annotations')},
        'allowed_annotation_changes':['q25'],
        'instruction':'This is one explicit source-backed editorial adjudication, not format repair or resampling until pass. Reassess q25 only under the supplied source scope and clearly marked new editorial intent. Inspect ALL original q25 issues, including secondary naturalness/tense concerns. Copy the other 29 annotations exactly, in order. Return the full response schema with your actual verdict and q25-prefixed remaining issues. Do not infer pass from the clarification or from any other reviewer: no other reviewer conclusions are supplied. No student material or blueprint may change.'}


def verified_annotation_packet(run, entry, packet, *, seen=None, return_payload=False):
    """Authenticate original/repair response chains before checking formatting."""
    uid, sha = entry['id'], entry['content_sha256']
    if packet.get('unit_id') != uid or packet.get('content_sha256') != sha:
        raise ValueError('Stale annotation binding')
    payload = fingerprint_payload(run, entry, input_version=packet.get('editorial_input_version', LEGACY_INPUT_VERSION))
    previous = None
    if 'format_repair_source' in packet and 'adjudication_source' in packet:
        raise ValueError('Format repair must inherit, not duplicate, adjudication provenance')
    if 'format_repair_source' in packet:
        source = packet['format_repair_source']
        previous, seen = archived_packet(run,source,seen)
        prior_payload = verified_annotation_packet(run,entry,previous,seen=seen,return_payload=True)
        errors = annotation_errors(entry,previous)
        correction = format_correction(entry,previous,errors)
        if packet.get('response_format_correction')!=correction:
            raise ValueError('Stored annotation format correction differs from its exact historical source')
        if packet.get('editorial_input_version',LEGACY_INPUT_VERSION)!=previous.get('editorial_input_version',LEGACY_INPUT_VERSION):
            raise ValueError('A narrow format repair cannot change the editorial input contract')
        payload={**{k:v for k,v in prior_payload.items() if k!='response_format_correction'},'response_format_correction':correction}
    elif 'response_format_correction' in packet:
        raise ValueError('Format correction has no authenticated original annotation')
    elif 'adjudication_source' in packet:
        source = packet['adjudication_source']
        previous, seen = archived_packet(run,source,seen)
        prior_payload = verified_annotation_packet(run,entry,previous,seen=seen,return_payload=True)
        if 'editorial_adjudication' in prior_payload:
            raise ValueError('Repeated semantic adjudication is prohibited, including through format wrappers')
        if packet.get('editorial_input_version',LEGACY_INPUT_VERSION) != previous.get('editorial_input_version',LEGACY_INPUT_VERSION):
            raise ValueError('Adjudication cannot rewrite the original base input version')
        context = adjudication_context(run,entry,previous,source)
        payload={**payload,'editorial_adjudication':context}
        check_adjudicated_changes(previous,packet)
    result = validate_api_evidence(run, read(run/packet['api_evidence']), 'fingerprint_reviewer', payload, FingerprintReview)
    if any(packet.get(k) != v for k, v in result.model_dump().items()):
        raise ValueError('Annotation sidecar differs from API result')
    if previous is not None and 'format_repair_source' in packet:
        check_narrow_changes(previous,packet,packet['response_format_correction'])
    return payload if return_payload else packet


def load_annotation(run, entry):
    uid,sha=entry['id'],entry['content_sha256']
    packet=read(run/'annotations'/f'{uid}-{sha[:12]}.reviewed.json')
    verified_annotation_packet(run,entry,packet)
    if errors:=annotation_errors(entry,packet):
        raise AnnotationValidationError(uid,errors)
    return packet


def repair_annotation_format(run,entry,*,max_calls=3,timeout=900):
    uid,sha=entry['id'],entry['content_sha256']
    path=run/'annotations'/f'{uid}-{sha[:12]}.reviewed.json'
    previous=read(path)
    # Hash/provenance failures stop here, before any new model call.
    prior_payload = verified_annotation_packet(run,entry,previous,return_payload=True)
    correction=format_correction(entry,previous,annotation_errors(entry,previous))
    raw=path.read_bytes()
    original_hash=hashlib.sha256(raw).hexdigest()
    archived=run/'annotations/history'/f'{uid}-{sha[:12]}.invalid-{original_hash[:12]}.json'
    archived.parent.mkdir(parents=True,exist_ok=True)
    if archived.exists() and archived.read_bytes()!=raw:
        raise ValueError('Conflicting immutable invalid-annotation archive')
    if not archived.exists():
        archived.write_bytes(raw)
    version=previous.get('editorial_input_version',LEGACY_INPUT_VERSION)
    payload={**{k:v for k,v in prior_payload.items() if k!='response_format_correction'},'response_format_correction':correction}
    result,evidence=api_call(run,'fingerprint_reviewer',payload,FingerprintReview,max_calls=max_calls,timeout=timeout)
    packet={'unit_id':uid,'content_sha256':sha,'model':ANTHROPIC_MODEL,'effort':EFFORT,
        'execution':'anthropic_api','api_evidence':evidence,'editorial_input_version':version,
        'format_repair_source':{'annotation_archive':str(archived.relative_to(run)),'annotation_sha256':original_hash},
        'response_format_correction':correction,**result.model_dump()}
    attempted=run/'annotations/format-repair-attempts'/f'{uid}-{sha[:12]}-{digest(evidence)[:12]}.json'
    save_json(attempted,packet)
    verified_annotation_packet(run,entry,packet)
    if errors:=annotation_errors(entry,packet):
        raise AnnotationValidationError(uid,errors)
    save_json(path,packet)
    print(f'{uid}: explicit API annotation-format repair complete; original semantic verdict remains {packet["verdict"]}',flush=True)
    return packet


def adjudicate_fingerprint(run, *, timeout=900):
    """Exactly one semantic call for this content; all failures consume it."""
    entry = next((e for e in selected_entries(run) if e['id']=='u001'),None)
    if entry is None:
        raise ValueError('Missing selected Part 5 checkpoint')
    contract_path, reservation_path = adjudication_paths(entry)
    if (run/reservation_path).exists():
        raise ValueError('One-time adjudication already reserved; no second API call allowed')
    path = run/'annotations'/f"u001-{entry['content_sha256'][:12]}.reviewed.json"
    previous = read(path)
    prior_payload = verified_annotation_packet(run,entry,previous,return_payload=True)
    if 'editorial_adjudication' in prior_payload:
        raise ValueError('Repeated semantic adjudication is prohibited')
    contract = adjudication_contract(run,entry)
    raw = path.read_bytes(); original_hash = hashlib.sha256(raw).hexdigest()
    archive = Path('annotations/history')/f"u001-{entry['content_sha256'][:12]}.pre-adjudication-{original_hash[:12]}.json"
    (run/archive).parent.mkdir(parents=True,exist_ok=True)
    if (run/archive).exists() and (run/archive).read_bytes()!=raw:
        raise ValueError('Conflicting immutable pre-adjudication archive')
    if not (run/archive).exists():
        (run/archive).write_bytes(raw)
    if (run/contract_path).exists() and read(run/contract_path)!=contract:
        raise ValueError('Conflicting immutable adjudication contract')
    if not (run/contract_path).exists():
        save_json(run/contract_path,contract)
    source={'annotation_archive':str(archive),'annotation_sha256':original_hash,
            'contract':str(contract_path),'contract_sha256':hashlib.sha256((run/contract_path).read_bytes()).hexdigest()}
    reservation={'protocol':ADJUDICATION_VERSION,'unit_id':'u001','content_sha256':entry['content_sha256'],'question_id':'q25','source':dict(source)}
    (run/reservation_path).parent.mkdir(parents=True,exist_ok=True)
    with (run/reservation_path).open('x') as handle:
        handle.write(json.dumps(reservation,ensure_ascii=False,indent=2)+'\n')
    source.update(reservation=str(reservation_path),reservation_sha256=hashlib.sha256((run/reservation_path).read_bytes()).hexdigest())
    context=adjudication_context(run,entry,previous,source)
    payload={**fingerprint_payload(run,entry,input_version=previous.get('editorial_input_version',LEGACY_INPUT_VERSION)),
             'editorial_adjudication':context}
    result,evidence=api_call(run,'fingerprint_reviewer',payload,FingerprintReview,max_calls=1,timeout=timeout)
    packet={'unit_id':'u001','content_sha256':entry['content_sha256'],'model':ANTHROPIC_MODEL,'effort':EFFORT,
            'execution':'anthropic_api','api_evidence':evidence,
            'editorial_input_version':previous.get('editorial_input_version',LEGACY_INPUT_VERSION),
            'adjudication_source':source,**result.model_dump()}
    save_json(run/'annotations/adjudications'/f"u001-{entry['content_sha256']}.result.json",packet)
    verified_annotation_packet(run,entry,packet)
    # A malformed q25 field remains visible for explicit narrow format repair;
    # semantic changes to the locked 29 never reach this replacement.
    save_json(path,packet)
    load_annotation(run,entry)
    print(f'u001/q25: one actual API adjudication returned {result.verdict}; original revise preserved',flush=True)
    return packet


def entries_and_annotations(run):
    entries = selected_entries(run)
    annotations = {}
    for entry in entries:
        path = run/'annotations'/f"{entry['id']}-{entry['content_sha256'][:12]}.reviewed.json"
        if path.exists():
            packet = load_annotation(run, entry)
            annotations[entry['id']] = packet['annotations']
    return entries, annotations


def diagnostic(run, entries, annotations, exceptions=None):
    config = configuration(run)
    result = DiversityHistory(run/'audit/history-before.json').audit_test(entries, config['run_id'], annotations, exceptions or {})
    metrics = whole_test_metrics(entries)
    result['whole_test_metrics'] = metrics
    result['policy'] = {'part7_longest_option_alert':.30,'part7_shortest_option_alert':.30,
        'purpose_planning_range':[5,8],'multiple_table_group_budget':3,
        'status':'Editorial alerts, not official TOEIC quotas or psychometric thresholds.'}
    alerts = []
    for entry in entries:
        plan = config['plans'][int(entry['id'][1:])-1]
        tables = sum(b['type']=='table' for d in entry['content']['documents'] for b in d['blocks'])
        if plan.get('table_policy') and tables > plan['table_policy']['max_table_blocks']:
            alerts.append(f"{entry['id']}: {tables} table blocks exceed planned maximum {plan['table_policy']['max_table_blocks']}")
    p7 = result['option_length']['parts'].get('7', {})
    for name in ('longest','shortest'):
        accuracy = p7.get(f'{name}_strategy_expected_accuracy')
        if accuracy is not None and accuracy > .30:
            alerts.append(f'Part 7 {name}-option strategy exceeds editorial alert .30')
    if metrics['p7_multiple_with_tables'] > 3:
        alerts.append('More than three multiple groups use tables')
    if metrics['p7_groups'] == 15 and not 5 <= metrics['p7_kind_counts'].get('purpose',0) <= 8:
        alerts.append('Purpose count outside declared planning range')
    result['editorial_alerts'] = alerts
    return result


def whole_adjudication_context(run, entries, version=WHOLE_ADJUDICATION_VERSION):
    if version != WHOLE_ADJUDICATION_VERSION:
        raise ValueError('Unknown whole-review adjudication context version')
    records=[]
    for entry in entries:
        path=run/'annotations'/f"{entry['id']}-{entry['content_sha256'][:12]}.reviewed.json"
        packet=load_annotation(run,entry)
        actual_payload=verified_annotation_packet(run,entry,packet,return_payload=True)
        context=actual_payload.get('editorial_adjudication')
        if context is None:
            continue
        records.append({'unit_id':entry['id'],'content_sha256':entry['content_sha256'],'question_id':'q25',
            'current_annotation_packet':str(path.relative_to(run)),
            'current_annotation_packet_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source':context['source'],'contract':context['contract'],
            'original_verdict':context['previous_response']['verdict'],'original_issues':context['previous_response']['issues'],
            'current_api_verdict':packet['verdict'],'current_api_issues':packet['issues'],
            'current_api_evidence':packet['api_evidence']})
    return {'version':version,'records':records,
        'instruction':'These are explicit, source-backed editorial-scope adjudications, not retroactive changes to old findings. Apply the source scope and clearly marked editorial intent when judging the whole test. Independently reassess actual construct necessity, naturalness, distractors and explanations; do not defer to the fingerprint outcome. Real materially different operations or weak/unnatural content still require revise. No external reviewer conclusions are supplied.'}


def whole_payload(run, entries, annotations, *, input_version=INPUT_VERSION, adjudication_context_version=None):
    config = configuration(run)
    payload = {'content_bindings':[{'unit_id':e['id'],'content_sha256':e['content_sha256']} for e in entries],
        'units':[{'unit_id':e['id'],'blueprint':config['plans'][int(e['id'][1:])-1], 'content':e['content'],
                  'annotations':annotations.get(e['id'],[])} for e in entries],
        'quantitative_audit':diagnostic(run, entries, annotations),
        'display_numbering':'Begin at 101, use unit/question order; Part 5 101-130, Part 6 131-146, Part 7 147-200.'}
    if adjudication_context_version is not None:
        payload['fingerprint_editorial_adjudications']=whole_adjudication_context(run,entries,adjudication_context_version)
    return with_input_configuration(run, payload, input_version)


def whole_review(run, *, max_calls=3, timeout=900):
    freeze_input_configuration(run)
    entries, annotations = entries_and_annotations(run)
    if len(entries) != 20 or set(annotations) != {e['id'] for e in entries}:
        raise ValueError('Whole-test API review requires all 20 accepted units and reviewed fingerprints')
    for entry in entries:
        packet = load_annotation(run, entry)
        if packet['verdict'] != 'pass' or packet['issues']:
            raise ValueError(f"{entry['id']}: unresolved independent fingerprint review")
    context = whole_adjudication_context(run,entries)
    context_version = WHOLE_ADJUDICATION_VERSION if context['records'] else None
    payload = whole_payload(run, entries, annotations,adjudication_context_version=context_version)
    result, evidence = api_call(run, 'whole_test_reviewer', payload, WholeReview, max_calls=max_calls, timeout=timeout)
    record = {'model':ANTHROPIC_MODEL,'effort':EFFORT,'execution':'anthropic_api',
              'api_evidence':evidence, 'editorial_input_version':INPUT_VERSION, **result.model_dump()}
    if context_version is not None:
        record['fingerprint_adjudication_context_version']=context_version
    save_json(run/'audit/whole-test-review.json', record)
    print(f'Whole-test API review: {result.verdict}', flush=True)
    return record


def audit(run, require_pass=False):
    entries, annotations = entries_and_annotations(run)
    exceptions = {}
    review = None
    review_path = run/'audit/whole-test-review.json'
    if review_path.exists():
        review = read(review_path)
        result = validate_api_evidence(run, read(run/review['api_evidence']), 'whole_test_reviewer',
                                      whole_payload(run, entries, annotations, input_version=review.get('editorial_input_version', LEGACY_INPUT_VERSION),
                                                    adjudication_context_version=review.get('fingerprint_adjudication_context_version')), WholeReview)
        if any(review.get(k) != v for k,v in result.model_dump().items()):
            raise ValueError('Whole-test sidecar differs from API result')
        if result.content_bindings != [ContentBinding(unit_id=e['id'],content_sha256=e['content_sha256']) for e in entries]:
            raise ValueError('Whole-test review bindings are incomplete or stale')
        reviewer_id = f"anthropic-api:{digest(read(run/review['api_evidence']))[:16]}:whole-test"
        for row in result.core_skill_reuse:
            if row.unit_id not in {e['id'] for e in entries}:
                raise ValueError('Core-skill reuse disposition names an unknown unit')
            exceptions.setdefault(row.unit_id,[]).append({'question_id':row.question_id,'finding_id':row.finding_id,
                'reviewer_id':reviewer_id,'reason':row.reason})
    report = diagnostic(run, entries, annotations, exceptions)
    if require_pass:
        if len(entries)!=20 or report['whole_test_metrics']['question_counts']!={'5':30,'6':16,'7':54}:
            raise ValueError('Final API diversity gate requires a complete 100-question test')
        for entry in entries:
            packet = load_annotation(run, entry)
            if packet['verdict']!='pass' or packet['issues']:
                raise ValueError(f"{entry['id']}: unresolved independent fingerprint findings")
        if not review or review['verdict']!='pass' or review['unresolved_issues'] or any(f['decision']=='revise' for f in review['findings']):
            raise ValueError('Missing passing exact-version Claude API whole-test review')
        resolved = [r['alert'] for r in review['resolved_alerts']]
        if len(resolved)!=len(set(resolved)) or set(resolved)!=set(report['editorial_alerts']):
            raise ValueError('Every whole-test alert needs an exact API editorial disposition')
        if report['review_required']:
            raise ValueError('Unresolved diversity findings')
        save_accepted_history(DiversityHistory(run/'audit/history-before.json'), entries,
                              report['edition_id'], annotations, exceptions, run/'audit/history-after.json')
        save_json(run/'audit/diversity-exceptions.json', exceptions)
    if review:
        report['whole_test_review'] = review
    save_json(run/'audit/diversity-audit.json', report)
    print(json.dumps({'units':len(entries),'annotation_units':len(annotations),
        'review_required':report['review_required'],'editorial_alerts':report['editorial_alerts']},ensure_ascii=False),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init')
    p.add_argument('--seed', type=int)
    p.add_argument('--corpus', type=Path, help='Optional external comparison corpus with ten test JSON files')
    p.add_argument('--run-id')
    p = sub.add_parser('annotate')
    p.add_argument('--units', nargs='+')
    p.add_argument('--max-api-calls', type=int, default=3)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--repair-invalid',action='store_true',help='One explicit API retry for narrowly eligible annotation formatting/quotation errors; never retry semantic revise or provenance failure')
    p = sub.add_parser('adjudicate-q25',help='One explicit source-backed editorial adjudication; original findings preserved, all other annotations locked')
    p.add_argument('--timeout',type=int,default=900)
    p = sub.add_parser('whole-review')
    p.add_argument('--max-api-calls', type=int, default=3)
    p.add_argument('--timeout', type=int, default=900)
    p = sub.add_parser('audit')
    p.add_argument('--require-pass', action='store_true')
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.run_dir, args.seed, args.corpus, run_id=args.run_id)
    elif args.command == 'annotate':
        annotate(args.run_dir.resolve(), args.units, max_calls=args.max_api_calls, timeout=args.timeout, repair_invalid=args.repair_invalid)
    elif args.command == 'whole-review':
        whole_review(args.run_dir.resolve(), max_calls=args.max_api_calls, timeout=args.timeout)
    elif args.command == 'adjudicate-q25':
        adjudicate_fingerprint(args.run_dir.resolve(),timeout=args.timeout)
    else:
        audit(args.run_dir.resolve(), args.require_pass)


if __name__ == '__main__':
    main()
