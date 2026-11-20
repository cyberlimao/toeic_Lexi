"""Prepare an explicit, provenance-preserving revision from external review.

Only qualitative difficulty metadata can be changed locally. Actual question or
explanation edits are delegated to the saved provider by the subsequent Engine resume call.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from toeic_engine.cli import initial_author_provenance
from toeic_engine.engine import content_hash, prompt_version, save_json, verify_entry
from toeic_engine.models import Unit
from toeic_engine.novelty import History


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--unit',required=True)
    p.add_argument('--feedback',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--explanations-only',action='store_true')
    p.add_argument('--questions')
    p.add_argument('--difficulty',action='append',default=[],help='Explicit qualitative recalibration, e.g. q3=medium')
    args=p.parse_args()
    if args.questions and not args.explanations_only: raise ValueError('--questions requires --explanations-only')
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Use a fresh output directory')
    source=args.source.resolve(); out=args.out.resolve()
    original=json.loads((source/'run.json').read_text())
    checkpoint=source/'units'/f'{args.unit}.json'
    entry=json.loads(checkpoint.read_text())
    if errors:=verify_entry(entry): raise ValueError(errors)
    unit=Unit.model_validate(entry['content']); original_digest=content_hash(unit)
    plan=copy.deepcopy(entry['blueprint']); adjustments=[]
    for change in args.difficulty:
        qid,level=change.split('=',1)
        if level not in ('easy','medium','hard'): raise ValueError('Invalid difficulty')
        question=next(q for q in unit.questions if q.id==qid)
        slot=next(s for s in plan['slots'] if s['id']==qid)
        adjustments.append({'question_id':qid,'from':question.difficulty,'to':level})
        question.difficulty=level;slot['difficulty']=level
    history=History(Path(original['history']))
    try: parent=history.revision_parent(f'{source}/{args.unit}',original_digest,original.get('revises'))
    finally: history.close()
    run={'run_id':out.name,'created_at':datetime.now(timezone.utc).isoformat(),
         'seed':original['seed'],'prompt_version':prompt_version(),'part':'audit',
         'history':original['history'],'corpus':original.get('corpus'),'plans':[plan],
         'author_provider':original.get('author_provider','gemini'),
         'review_provider':original.get('review_provider','gemini'),
         'initial_author_provenance':initial_author_provenance(entry, original, False),
         'initial_draft':unit.model_dump(),'initial_feedback':[args.feedback.read_text()],
         'revises':parent,'audited_draft_source':str(checkpoint),
         'explanation_only':args.explanations_only,
         'repair_questions':args.questions.split(',') if args.questions else None,
         'editorial_preparation':{'base_content_sha256':original_digest,
             'difficulty_recalibration':adjustments,'feedback_source':str(args.feedback.resolve()),
             'note':'Only explicit difficulty metadata adjusted locally; student content and keys preserved before model revision.'}}
    save_json(out/'run.json',run)
    print(f'Prepared {out}; resume with toeic_engine to revise and run all quality gates.')


if __name__=='__main__': main()
