"""Ask the source run's author for bounded edits, then prepare a re-audited run.

This tool does not publish content. Run `toeic_engine resume OUTPUT` afterwards.
Unlisted student fields are preserved byte-for-byte; explanations are revised next.
"""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from toeic_engine.anthropic_provider import AnthropicProvider
from toeic_engine.openai_provider import OpenAIProvider
from toeic_engine.cli import initial_author_provenance
from toeic_engine.engine import content_hash, model_provenance, prompt_version, save_json
from toeic_engine.models import Unit
from toeic_engine.novelty import History
from toeic_engine.provider import VertexProvider


class Change(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str
    value: str


class Changes(BaseModel):
    model_config = ConfigDict(extra='forbid')
    changes: list[Change]


def location(data, path):
    if not path.startswith('/'):
        raise ValueError('Paths must start with /')
    parts = path[1:].split('/')
    parent = data
    for key in parts[:-1]:
        parent = parent[int(key)] if isinstance(parent, list) else parent[key]
    key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
    if not isinstance(parent[key], str):
        raise ValueError('Only existing string fields can be revised')
    return parent, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--unit', default='u001', help='Source run unit ID; default u001')
    source_mode = parser.add_mutually_exclusive_group()
    source_mode.add_argument('--attempt', type=int, help='Rejected attempt number; omit to use the accepted checkpoint')
    source_mode.add_argument('--initial-draft', action='store_true', help='Use a previously prepared run.json initial_draft; never marks it accepted')
    parser.add_argument('--feedback', type=Path, required=True)
    parser.add_argument('--paths', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--questions', help='Comma-separated question IDs whose explanations need revision; default all')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new output directory')
    original = json.loads((args.source / 'run.json').read_text())
    if not args.unit.startswith('u') or not args.unit[1:].isdigit():
        raise ValueError('Invalid unit ID')
    unit_index = int(args.unit[1:]) - 1
    if not 0 <= unit_index < len(original['plans']):
        raise ValueError('Unit is outside this run')
    if args.attempt is not None and args.attempt < 1:
        raise ValueError('Attempt must be positive')
    source_path = (args.source / 'run.json' if args.initial_draft else
                   args.source / 'attempts' / f'{args.unit}-a{args.attempt}.json'
                   if args.attempt is not None else args.source / 'units' / f'{args.unit}.json')
    raw = json.loads(source_path.read_text())
    if args.initial_draft:
        if unit_index != 0 or len(original['plans']) != 1:
            raise ValueError('Prepared initial drafts require a single-unit source run')
        data = raw['initial_draft']
        source_author = initial_author_provenance({'provenance': raw.get('initial_author_provenance', {})}, original, False)
    else:
        data = raw if args.attempt is not None else raw['content']
        source_author = initial_author_provenance(raw, original, args.attempt is not None)
    unit = Unit.model_validate(data)
    if len(set(args.paths)) != len(args.paths):
        raise ValueError('Duplicate paths')
    current = {path: location(data, path)[0][location(data, path)[1]] for path in args.paths}
    feedback = args.feedback.read_text()
    api = args.out.resolve() / 'api' / 'targeted-preparation'
    choices = {role: original.get(role, 'gemini') for role in ('author_provider', 'review_provider')}
    providers = {'gemini': VertexProvider, 'claude': AnthropicProvider, 'openai': OpenAIProvider}
    if any(choice not in providers for choice in choices.values()):
        raise ValueError('Providers must be gemini, claude or openai')
    # This custom editing prompt is author work. Do not route it as a review.
    provider = providers[choices['author_provider']](api, max_calls=3, timeout_seconds=900)
    preparation_author = model_provenance(provider)
    try:
        edits = provider.generate(
            'You are a precise TOEIC English editor. Apply the supplied expert feedback ONLY to the listed string paths. Return every path once in the supplied order, with its complete replacement value. Preserve all other text by not returning it. Follow concrete proposed replacements closely. Do not add new facts, policies, actors or conditions beyond the requested repair. Keep IDs, keys and unaffected content unchanged. Explanations will be repaired separately; do not return explanations. The final complete group will undergo fresh blind and editorial review.',
            {'unit':data, 'allowed_paths_and_current_values':current, 'required_fixes':feedback}, Changes, 'text-edits', max_tokens=16384)
    finally:
        provider.close()
    if [c.path for c in edits.changes] != args.paths:
        raise ValueError('Author changed the requested path set/order')
    revised = copy.deepcopy(data)
    for change in edits.changes:
        parent, key = location(revised, change.path)
        parent[key] = change.value
    revised = Unit.model_validate(revised)
    history = History(Path(original['history']))
    try:
        revises = history.revision_parent(f'{args.source.resolve()}/{args.unit}', content_hash(unit), original.get('revises'))
    finally:
        history.close()
    run = {'run_id':args.out.name, 'created_at':datetime.now(timezone.utc).isoformat(),
           'seed':original['seed'], 'prompt_version':prompt_version(), 'part':'audit',
           'history':original['history'], 'corpus':original.get('corpus'), 'plans':[original['plans'][unit_index]],
           'initial_draft':revised.model_dump(), 'initial_feedback':[feedback],
           'initial_author_provenance':source_author, **choices,
           'revises':revises, 'explanation_only':True, 'repair_questions':args.questions.split(',') if args.questions else None,
           'editorial_preparation':{**preparation_author, 'base_content_sha256':content_hash(unit),
                                    'source_artifact':str(source_path),
                                    **({'source_status': 'prepared_unaccepted_initial_draft',
                                        'source_pointer': '/initial_draft',
                                        'source_artifact_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}
                                       if args.initial_draft else {}),
                                    'changed_paths':args.paths, 'api_directory':str(api)}}
    save_json(args.out / 'run.json', run)
    print(f'Prepared {args.out}; run python -m toeic_engine resume {args.out} to revise explanations and re-audit.')


if __name__ == '__main__':
    main()
