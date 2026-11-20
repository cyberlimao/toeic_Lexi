"""Resume an all-Claude API run with auditable exact-request review replay.

Only completed, byte-bound API evidence may supply a review response. Author
and explanation-editor requests are always sent to Claude. Replay selection
uses the earliest completed response artifact in a frozen deterministic index,
never its verdict. This program does not relax any Engine acceptance gate.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import uuid

from pydantic import ValidationError

from toeic_engine.anthropic_provider import AnthropicProvider, ANTHROPIC_MODEL, EFFORT, anthropic_schema
from toeic_engine.cli import export
from toeic_engine.engine import Engine, PROMPT_NAMES, prompt, prompt_version, save_json, verify_entry
from toeic_engine.independent_audits import EvidenceDecision, TeachingReview, digest, evidence_schema, teaching_schema
from toeic_engine.models import BlindSolution, Challenge, ExplanationRevision, Review, Unit
from toeic_engine.novelty import History
from toeic_engine.provider import OutputError, ProviderError

PROTOCOL = 'claude-exact-request-review-replay-v1'
SCHEMAS = {'author':Unit,'solver':BlindSolution,'challenger':Challenge,'critic':Review,
           'evidence_probe':EvidenceDecision,'teaching_reviewer':TeachingReview,'explanation_editor':ExplanationRevision}
REPLAY_ROLES = {'solver','challenger','critic','evidence_probe','teaching_reviewer'}
CLAUDE = {'model':ANTHROPIC_MODEL,'effort':EFFORT,'thinking':'adaptive','execution':'anthropic_api'}


def allowed_role_schema(role,payload,schema):
    """Accept only core schemas or the two official question-bound factories.

    Their API wire schema is deliberately identical to the base model, but
    local validators bind the actual requested question IDs. Preserve those
    validators instead of accepting arbitrary compatible subclasses.
    """
    base=SCHEMAS[role]
    if schema is base:
        return True
    if role=='teaching_reviewer':
        qids=[q['id'] for q in payload.get('questions',[])]
        if not 1<=len(qids)<=5 or len(set(qids))!=len(qids):
            return False
        expected=teaching_schema(qids)
        validator_name='requested_questions'
    elif role=='evidence_probe':
        qid=payload.get('question',{}).get('id')
        if not isinstance(qid,str) or not qid:
            return False
        expected=evidence_schema(qid)
        validator_name='requested_question'
    else:
        return False
    if (not isinstance(schema,type) or not issubclass(schema,base)
            or schema.__module__!=expected.__module__ or schema.__qualname__!=expected.__qualname__
            or schema.model_json_schema()!=expected.model_json_schema()):
        return False
    actual=schema.__pydantic_decorators__.model_validators.get(validator_name)
    wanted=expected.__pydantic_decorators__.model_validators[validator_name]
    if actual is None or actual.func.__code__ is not wanted.func.__code__:
        return False
    actual_binding=tuple(c.cell_contents for c in (actual.func.__closure__ or ()))
    wanted_binding=tuple(c.cell_contents for c in (wanted.func.__closure__ or ()))
    return actual_binding==wanted_binding


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wire_request(system, payload, schema, max_tokens):
    return {'model':ANTHROPIC_MODEL,'max_tokens':max(65536,max_tokens),
            'thinking':{'type':'adaptive'},'stream':True,'system':system,
            'messages':[{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
            'output_config':{'effort':EFFORT,'format':{'type':'json_schema','schema':anthropic_schema(schema)}}}


def identity(role, wire):
    return {'role':role,'wire_request':wire}


def check_request_record(record, role, prompts):
    """Validate self-consistency before treating a persisted wire as evidence."""
    if role not in REPLAY_ROLES or record.get('provider')!='anthropic':
        raise ValueError('Not a replayable Anthropic review role')
    if record.get('model')!=ANTHROPIC_MODEL or record.get('effort')!=EFFORT:
        raise ValueError('Source request model/effort mismatch')
    if record.get('system')!=prompts[role] or record.get('schema')!=anthropic_schema(SCHEMAS[role]):
        raise ValueError('Source role/system/schema mismatch')
    wire = record.get('wire_request',{})
    expected = wire_request(record['system'],record['payload'],SCHEMAS[role],wire.get('max_tokens',0))
    if wire!=expected:
        raise ValueError('Source wire differs from exact model/system/payload/schema/max_tokens contract')
    return wire


def completed_source(usage_path, row_index, row, role, prompts):
    folder = usage_path.parent
    request_path = folder/f"{row['label']}.request.json"
    response_path = folder/f"{row['label']}.response.txt"
    record = read(request_path)
    wire = check_request_record(record,role,prompts)
    return {
        'role':role,'request':str(request_path.resolve()),'response':str(response_path.resolve()),
        'usage':str(usage_path.resolve()),'usage_row_index':row_index,'usage_row_sha256':digest(row),
        'request_sha256':file_hash(request_path),'response_sha256':file_hash(response_path),
        'response_mtime_ns':response_path.stat().st_mtime_ns,
        'request_identity_sha256':digest(identity(role,wire)),
    }


def source_index(roots, prompts):
    """Freeze complete evidence; ongoing requests are never candidates.

    usage.json may grow as another worker appends calls. We bind its specific
    completed row rather than require the whole append-only file to stay fixed.
    """
    candidates, seen, incomplete_snapshots = [], set(), []
    systems = {prompts[role]:role for role in REPLAY_ROLES}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Replay source root does not exist: {root}')
        for usage_path in sorted(root.rglob('usage.json')):
            usage_path = usage_path.resolve()
            if usage_path in seen:
                continue
            seen.add(usage_path)
            try:
                rows = read(usage_path)
            except json.JSONDecodeError:
                # An active provider writes its growing usage ledger. A
                # partially written snapshot is unavailable, not replay proof.
                incomplete_snapshots.append(str(usage_path))
                continue
            if not isinstance(rows,list):
                continue
            for row_index,row in enumerate(rows):
                if not isinstance(row,dict) or not row.get('stream_complete') or row.get('finish_reason')!='end_turn':
                    continue
                request_path = usage_path.parent/f"{row['label']}.request.json"
                if not request_path.exists():
                    raise ValueError(f'Completed usage has no source request: {request_path}')
                record = read(request_path)
                role = systems.get(record.get('system'))
                if role is None or record.get('provider')!='anthropic':
                    continue
                candidates.append(completed_source(usage_path,row_index,row,role,prompts))
    candidates.sort(key=lambda c:(c['response_mtime_ns'],c['request'],c['usage_row_index']))
    return {'protocol':PROTOCOL,'recorded_at':datetime.now(timezone.utc).isoformat(),
        'selection_order':'Earliest response artifact mtime_ns in this frozen snapshot, then absolute request path, then usage row index; no result/verdict filtering.',
        'incomplete_usage_snapshots_not_used':incomplete_snapshots,'sources':candidates}


def validate_reuse_event(event, *, prompts=None):
    """Pure local validation for both the runner and release-report builder.

    Returns the original response text and bound usage row. It validates the
    evidence, not whether the model verdict should pass a content quality gate.
    """
    prompts = prompts or {name:prompt(name) for name in PROMPT_NAMES}
    if event.get('kind') not in ('exact_request_reuse','api_call') or event.get('protocol')!=PROTOCOL:
        raise ValueError('Unsupported replay event')
    if (event.get('model'),event.get('effort'))!=(ANTHROPIC_MODEL,EFFORT):
        raise ValueError('Replay event model/effort mismatch')
    role = event['role']
    if role not in REPLAY_ROLES:
        raise ValueError('Author/editor responses cannot be replayed')
    source = event['source']
    for field in ('request','response'):
        if file_hash(source[field])!=source[field+'_sha256']:
            raise ValueError(f'Replay source {field} hash mismatch')
    request = read(source['request'])
    wire = check_request_record(request,role,prompts)
    expected = event['requested_identity']
    if expected!=identity(role,wire) or digest(expected)!=event['request_identity_sha256']:
        raise ValueError('Replay source differs from the complete requested identity')
    if source.get('role')!=role or source.get('request_identity_sha256')!=event['request_identity_sha256']:
        raise ValueError('Replay source role/identity mismatch')
    if digest(request['payload'])!=event['input_sha256']:
        raise ValueError('Replay event input hash mismatch')
    rows = read(source['usage'])
    index = source['usage_row_index']
    if not isinstance(index,int) or index<0 or index>=len(rows) or digest(rows[index])!=source['usage_row_sha256']:
        raise ValueError('Replay source completed usage row changed or disappeared')
    row = rows[index]
    if (row.get('model'),row.get('model_version'),row.get('effort'),row.get('finish_reason'),row.get('stream_complete'))!=(
            ANTHROPIC_MODEL,ANTHROPIC_MODEL,EFFORT,'end_turn',True):
        raise ValueError('Replay source lacks completed exact-model API usage')
    if Path(source['request']).name!=f"{row.get('label')}.request.json" or Path(source['response']).name!=f"{row.get('label')}.response.txt":
        raise ValueError('Replay source label does not bind request/response')
    if row.get('content_sha256')!=source['response_sha256']:
        raise ValueError('Replay response does not match the completed API usage hash')
    return {'raw_response':Path(source['response']).read_text(),'usage_row':row,'wire_request':wire}


class ExactClaudeReplay:
    model = ANTHROPIC_MODEL
    reasoning = {'effort':EFFORT,'thinking':'adaptive'}
    author_provenance = CLAUDE
    review_provenance = {**CLAUDE,'review_response_mode':'fresh_api_or_documented_exact_request_reuse'}

    def __init__(self, folder, roots, max_calls=80, timeout=900, *, transport=None):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True,exist_ok=True)
        self.prompts = {name:prompt(name) for name in PROMPT_NAMES}
        self.roles = {system:name for name,system in self.prompts.items()}
        self.index = source_index(roots,self.prompts)
        save_json(self.folder/'source-index.json',self.index)
        self.events = []
        self._transport = transport
        self.max_calls,self.timeout = max_calls,timeout
        self._summary()

    @property
    def calls(self):
        return self._transport.calls if self._transport is not None else 0

    def _summary(self):
        save_json(self.folder/'events.json',self.events)
        save_json(self.folder/'summary.json',{'protocol':PROTOCOL,'model':ANTHROPIC_MODEL,'effort':EFFORT,
            'actual_api_calls_including_retries':self.calls,
            'exact_request_reuse_events':sum(e['kind']=='exact_request_reuse' for e in self.events),
            'fresh_request_events':sum(e['kind'] in ('api_call','api_call_error') for e in self.events),
            'schema_rejected_reuse_events':sum(e['kind']=='exact_request_reuse' and e.get('validation')=='schema_rejected' for e in self.events),
            'source_validation_errors':sum(e.get('validation')=='source_rejected' for e in self.events),
            'note':'Reused responses are prior actual API outputs, not fresh independent reviews. Selection is verdict-blind; all original quality gates remain active.'})

    def close(self):
        self._summary()
        if self._transport is not None:
            self._transport.close()

    def _register_completed_review(self,role,actual_label):
        """Retain complete schema-invalid responses too, without verdict bias."""
        usage_path=self.folder/'api/usage.json'
        if not usage_path.exists():
            return None
        rows=read(usage_path)
        matching=[(i,r) for i,r in enumerate(rows) if r.get('label')==actual_label and r.get('stream_complete') and r.get('finish_reason')=='end_turn']
        if not matching:
            return None
        i,row=matching[0]
        source=completed_source(usage_path,i,row,role,self.prompts)
        if source not in self.index['sources']:
            self.index['sources'].append(source)
            self.index['sources'].sort(key=lambda c:(c['response_mtime_ns'],c['request'],c['usage_row_index']))
            save_json(self.folder/'source-index.json',self.index)
        return source

    def generate(self, system, payload, schema, label, max_tokens=65536):
        role = self.roles.get(system)
        if role is None or not allowed_role_schema(role,payload,schema):
            raise ProviderError('Replay runner only accepts exact current Engine role prompts/schemas')
        requested = identity(role,wire_request(system,payload,schema,max_tokens))
        request_hash = digest(requested)
        event = {'protocol':PROTOCOL,'role':role,'requested_label':label,'model':ANTHROPIC_MODEL,'effort':EFFORT,
            'requested_identity':requested,'request_identity_sha256':request_hash,'input_sha256':digest(payload),
            'recorded_at':datetime.now(timezone.utc).isoformat()}
        candidates = [s for s in self.index['sources'] if s['request_identity_sha256']==request_hash] if role in REPLAY_ROLES else []
        if candidates:
            # Already sorted by frozen completion order; never inspect a
            # verdict to choose among multiple actual responses.
            event.update(kind='exact_request_reuse',source=copy.deepcopy(candidates[0]))
            try:
                evidence = validate_reuse_event(event,prompts=self.prompts)
            except Exception:
                event['validation']='source_rejected'
                self.events.append(event);self._summary()
                raise
            try:
                result = schema.model_validate_json(evidence['raw_response'])
            except ValidationError as exc:
                event['validation']='schema_rejected'
                self.events.append(event);self._summary()
                raise OutputError(str(exc)) from None
            event.update(validation='complete_original_response',returned_at=datetime.now(timezone.utc).isoformat())
            self.events.append(event);self._summary()
            return result
        if self._transport is None:
            self._transport = AnthropicProvider(self.folder/'api',self.max_calls,self.timeout)
        actual_label = f'{len(self.events)+1:04d}-{label}'
        before_calls = self.calls
        try:
            result = self._transport.generate(system,payload,schema,actual_label,max_tokens=max_tokens)
        except Exception as exc:
            event.update(kind='api_call_error',actual_api_label=actual_label,actual_api_attempts=self.calls-before_calls,
                         error_type=type(exc).__name__)
            if role in REPLAY_ROLES:
                try:
                    if source:=self._register_completed_review(role,actual_label):
                        event['source']=source
                except Exception as evidence_error:
                    event['source_registration_error']=type(evidence_error).__name__
            self.events.append(event);self._summary()
            raise
        event.update(kind='api_call',actual_api_label=actual_label,actual_api_attempts=self.calls-before_calls,
                     validation='complete_original_response',returned_at=datetime.now(timezone.utc).isoformat())
        if role in REPLAY_ROLES:
            try:
                source=self._register_completed_review(role,actual_label)
                if source is None:
                    raise ValueError('New review has no completed actual API evidence')
                event['source']=source
                validate_reuse_event(event,prompts=self.prompts)
            except Exception:
                event['validation']='source_rejected'
                self.events.append(event);self._summary()
                raise
        self.events.append(event);self._summary()
        return result


def resume(run:Path, roots:list[Path], *, max_calls=80, max_attempts=1, timeout=900):
    run=run.resolve()
    config=read(run/'run.json')
    if config.get('author_provider')!='claude' or config.get('review_provider')!='claude':
        raise ValueError('Exact replay resume requires an all-Claude API run')
    if config['prompt_version']!=prompt_version():
        raise ValueError('Role prompts changed after this run was prepared')
    invocation=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    folder=run/'replay-runs'/invocation
    with (run/'.exact-api-resume.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        provider=ExactClaudeReplay(folder,roots,max_calls,timeout)
        history=History(Path(config['history']))
        entries=[]
        try:
            engine=Engine(provider,history,run,max_attempts,lambda s:print(s,flush=True))
            for i,plan in enumerate(config['plans'],1):
                uid=f'u{i:03d}'
                checkpoint=run/'units'/f'{uid}.json'
                pending=run/'units'/f'{uid}.pending.json'
                if not checkpoint.exists() and pending.exists():
                    entry=read(pending)
                    if entry['blueprint']==plan and not verify_entry(entry) and history.has_accepted(f'{run}/{uid}',entry['content_sha256']):
                        pending.replace(checkpoint)
                if checkpoint.exists():
                    entry=read(checkpoint)
                    if entry['blueprint']!=plan or verify_entry(entry):
                        raise ValueError(f'Invalid accepted checkpoint {uid}')
                    print(f'{uid}: retained accepted checkpoint',flush=True)
                else:
                    entry=engine.generate_unit(plan,uid,Unit.model_validate(config['initial_draft']) if 'initial_draft' in config else None,
                        config.get('initial_feedback'),config.get('revises'),config.get('explanation_only',False),
                        config.get('repair_questions'),initial_author_provenance=config.get('initial_author_provenance'))
                    entry['provenance']['review_execution']={
                        'mode':PROTOCOL,'model':ANTHROPIC_MODEL,'effort':EFFORT,
                        'events':str((folder/'events.json').resolve()),'summary':str((folder/'summary.json').resolve()),
                        'note':'Each review request is either a fresh Claude API call or explicitly documented exact-input reuse of its earliest completed source; author/editor calls are always fresh.'}
                    if config.get('editorial_preparation'):
                        entry['provenance']['editorial_preparation']=config['editorial_preparation']
                    save_json(checkpoint,entry)
                entries.append(entry)
                export(run,config,entries)
        finally:
            provider.close();history.close()
            if entries:
                export(run,config,entries)
        print(json.dumps(read(folder/'summary.json'),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--source-root',type=Path,action='append',required=True)
    parser.add_argument('--max-api-calls',type=int,default=80)
    parser.add_argument('--max-attempts',type=int,default=1)
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args()
    if min(args.max_api_calls,args.max_attempts,args.timeout)<1:
        parser.error('Limits must be positive')
    resume(args.run,args.source_root,max_calls=args.max_api_calls,max_attempts=args.max_attempts,timeout=args.timeout)


if __name__=='__main__':
    main()
