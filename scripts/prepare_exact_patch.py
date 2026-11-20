"""Apply an explicit reviewer correction, then require fresh model quality gates.

Patch files record exact old/new strings, source hash and reviewer report. This
preparation never creates an accepted checkpoint. Resume with --max-attempts 1
to audit the supplied draft without another author rewrite.
"""
import argparse
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from toeic_engine.engine import content_hash, prompt_version, save_json, verify_entry
from toeic_engine.models import Unit
from toeic_engine.novelty import History
from scripts.prepare_targeted_revision import location


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--unit', required=True)
    p.add_argument('--attempt', type=int, help='Use a saved valid draft instead of an accepted checkpoint')
    p.add_argument('--patch', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-student-edit', action='store_true')
    p.add_argument('--difficulty', action='append', default=[])
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a fresh output directory')
    source = args.source.resolve()
    original = json.loads((source / 'run.json').read_text())
    if not re.fullmatch(r'u\d{3}', args.unit):
        raise ValueError('Invalid unit ID')
    if args.attempt:
        checkpoint = source / 'attempts' / f'{args.unit}-a{args.attempt}.json'
        draft = Unit.model_validate_json(checkpoint.read_text())
        entry = {'content': draft.model_dump(), 'content_sha256': content_hash(draft),
                 'blueprint': original['plans'][int(args.unit[1:]) - 1]}
    else:
        checkpoint = source / 'units' / f'{args.unit}.json'
        entry = json.loads(checkpoint.read_text())
        if errors := verify_entry(entry):
            raise ValueError(errors)
    patch = json.loads(args.patch.read_text())
    if patch['source_content_sha256'] != entry['content_sha256']:
        raise ValueError('Stale patch')
    if not patch['review_report'] or not patch['changes']:
        raise ValueError('Reviewer provenance and changes required')
    report = json.loads(Path(patch['review_report']).read_text())
    if (report.get('content_sha256'), report.get('model'), report.get('reasoning_effort')) != (entry['content_sha256'], 'gpt-6-astra', 'high'):
        raise ValueError('Patch must reference the matching Astra/high review')
    data = copy.deepcopy(entry['content'])
    seen = set()
    for change in patch['changes']:
        path = change['path']
        explanation = re.fullmatch(r'/questions/\d+/explanation/(summary_zh|reasoning_zh|takeaway_zh|option_analysis/\d+/reason_zh|evidence/\d+/quote)', path)
        evidence_list = re.fullmatch(r'/questions/(\d+)/explanation/evidence', path)
        document = re.fullmatch(r'/documents/\d+/blocks/\d+/(text|rows/\d+/\d+)', path)
        question_stem = re.fullmatch(r'/questions/\d+/prompt', path)
        option_text = re.fullmatch(r'/questions/\d+/options/[0-3]/text', path)
        scenario_metadata = path == '/scenario_summary'
        if not explanation and not evidence_list and not scenario_metadata and not ((document or question_stem or option_text) and args.allow_student_edit):
            raise ValueError(f'Unsupported editorial path: {path}')
        if path in seen:
            raise ValueError('Duplicate patch path')
        seen.add(path)
        if evidence_list:
            parent, key = data['questions'][int(evidence_list[1])]['explanation'], 'evidence'
            if not isinstance(change['after'], list):
                raise ValueError('Evidence replacement must be a list')
        else:
            parent, key = location(data, path)
        if parent[key] != change['before']:
            raise ValueError(f'Old text mismatch: {path}')
        parent[key] = change['after']
    plan = copy.deepcopy(entry['blueprint'])
    adjustments = []
    for change in args.difficulty:
        qid, level = change.split('=', 1)
        if level not in ('easy', 'medium', 'hard'):
            raise ValueError('Invalid difficulty')
        question = next(q for q in data['questions'] if q['id'] == qid)
        slot = next(s for s in plan['slots'] if s['id'] == qid)
        adjustments.append({'question_id': qid, 'from': question['difficulty'], 'to': level})
        question['difficulty'] = slot['difficulty'] = level
    unit = Unit.model_validate(data)
    history = History(Path(original['history']))
    try:
        parent = history.revision_parent(f'{source}/{args.unit}', entry['content_sha256'], original.get('revises'))
    finally:
        history.close()
    save_json(args.out / 'run.json', {
        'run_id': args.out.name, 'created_at': datetime.now(timezone.utc).isoformat(),
        'seed': original['seed'], 'prompt_version': prompt_version(), 'part': 'audit',
        'history': original['history'], 'corpus': original.get('corpus'), 'plans': [plan],
        'initial_draft': unit.model_dump(), 'initial_feedback': [], 'revises': parent,
        'editorial_preparation': {
            'mode': 'exact_reviewed_patch',
            'editor': 'Codex applying explicit GPT-6-astra/high review findings',
            'base_content_sha256': entry['content_sha256'],
            'source_artifact': str(checkpoint),
            'source_status': 'unaccepted_draft' if args.attempt else 'accepted_automated',
            'prepared_content_sha256': content_hash(unit),
            'review_report': patch['review_report'], 'changes': patch['changes'],
            'difficulty_recalibration': adjustments,
            'note': 'Source test authored by Gemini; these precise editorial corrections are local. Fresh blind, challenge, critic and external review remain mandatory.'}})
    print(f'Prepared {args.out}; resume with --max-attempts 1 for fresh quality gates.')


if __name__ == '__main__':
    main()
