"""Prepare isolated student/full packets for an actual native external examiner."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from toeic_engine.engine import save_json, verify_entry
from toeic_engine.models import Unit, blind_payload
from scripts.native_reviewed_test import ROOT, reference
from scripts.release_reviewed_test import external_identity


def read(path):
    return json.loads(Path(path).read_text())


def prepare(run, name, units):
    run = run.resolve()
    metadata = read(run / 'run.json')
    override_path = run / 'audit/selected-checkpoints.json'
    overrides = read(override_path) if override_path.exists() else {}
    tasks = []
    for uid in units:
        index = int(uid[1:]) - 1
        plan = metadata['plans'][index]
        source = ROOT / overrides[uid] if uid in overrides else run / 'units' / f'{uid}.json'
        entry = read(source)
        if errors := verify_entry(entry):
            raise ValueError(f'{uid}: {errors}')
        if entry['blueprint'] != plan:
            raise ValueError(f'{uid}: undeclared blueprint change')
        unit = Unit.model_validate(entry['content'])
        number = 101 + sum(p['question_count'] for p in metadata['plans'][:index])
        common = {'unit_id': uid, 'content_sha256': entry['content_sha256'],
                  'display_numbers': {q.id: number + j for j, q in enumerate(unit.questions)}}
        key = f"{uid}-{entry['content_sha256'][:12]}"
        blind = run / 'external/packets' / f'{key}.blind.json'
        full = run / 'external/packets' / f'{key}.full.json'
        actual_plan = copy.deepcopy(plan)
        actual_plan.pop('cross_edition_editorial_context', None)
        for target, payload in [(blind, {**common, 'student_material': blind_payload(unit)}),
                                (full, {**common, 'content': unit.model_dump(), 'blueprint': actual_plan})]:
            if target.exists() and read(target) != payload:
                raise ValueError(f'Refusing to overwrite changed packet {target}')
            if not target.exists():
                save_json(target, payload)
        tasks.append({**common, 'blind_input': reference(blind), 'full_input': reference(full),
                      'blind_output': reference(run / 'external' / f'{key}.blind-review.json'),
                      'explanation_output': reference(run / 'external' / f'{key}.explanation-review.json'),
                      'review_output': reference(run / 'external' / f'{key}.review.json')})
    target = run / 'external/assignments' / f'{name}.json'
    protocol = run / 'external/protocol-v2.txt'
    if not protocol.exists():
        protocol = run / 'external/protocol.txt'
    if not protocol.exists():
        protocol = ROOT / 'toeic_engine/prompts/external_examiner.txt'
    external_model, external_effort = external_identity(metadata)
    value = {'assignment': name, 'required_model': external_model, 'reasoning_effort': external_effort,
             'protocol': reference(protocol), 'tasks': tasks,
             'boundary': 'Read only listed files, blind first. Do not read internal reviews, history, corrections, or source checkpoints.'}
    if target.exists() and read(target) != value:
        raise ValueError('Use a new assignment name for a changed task')
    save_json(target, value)
    print(reference(target))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--units', nargs='+', required=True)
    args = p.parse_args()
    prepare(args.run_dir, args.name, args.units)
