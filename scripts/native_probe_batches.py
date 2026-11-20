"""Pack independent native probes without exposing two views of the same unit.

Each batch contains at most one physically reduced question from each unit.
The required number of fresh contexts is the largest per-unit task count (16
for two cross-document questions in a triple set), regardless of slot order.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from scripts.native_reviewed_test import read, save_json, probes, check_probe, selected, selected_unit, freeze_instructions, native_identity
from toeic_engine.independent_audits import ablation_payload, document_subsets, digest


def pack(run: Path, prefix='independent-probe'):
    freeze_instructions(run)
    model,effort=native_identity(run)
    bins=[]
    for uid,spec in sorted(selected(run).items()):
        unit=selected_unit(spec)
        tasks=[]
        for q in unit.questions:
            if q.kind!='cross_document':continue
            for visible in document_subsets(unit):
                payload=ablation_payload(unit,q.id,visible)
                tasks.append({'unit_id':uid,'content_sha256':spec['content_sha256'],'question_id':q.id,
                              'visible_document_ids':visible,'input_sha256':digest(payload),'payload':payload})
        while len(bins)<len(tasks):bins.append([])
        for i,task in enumerate(tasks):bins[i].append(task)
    for i,items in enumerate(bins,1):
        name=f'{prefix}-{i:02d}'
        assert len({p['unit_id'] for p in items})==len(items)
        inp=f'packets/{name}.json';out=f'reviews/probe-batches/{name}.json'
        save_json(run/inp,items)
        save_json(run/'assignments'/f'{name}.json',{
            'role':'probe','model':model,'reasoning_effort':effort,'input':inp,'output':out,
            'assigned_units':[p['unit_id'] for p in items],
            'instruction_file':'instructions/evidence_probe.txt','protocol_file':'native-review-protocol.txt','schema_file':'probe.schema.json',
            'boundary':'One question and one fixed visible subset per unit, in a fresh context. Read only this packet and listed instructions/schema. No other packets, canonical units, plans, keys or prior reviews.'})
    save_json(run/'audit'/f'{prefix}-batches.json',{'names':[f'{prefix}-{i:02d}' for i in range(1,len(bins)+1)],
              'tasks':sum(len(b) for b in bins),'contexts':len(bins),'method':'Disjoint unit identities inside each fresh native probe context.'})
    print(f'{sum(len(b) for b in bins)} physically reduced tasks in {len(bins)} fresh-context batches')


def merge(run: Path,prefix='independent-probe'):
    meta=read(run/'audit'/f'{prefix}-batches.json')
    groups=defaultdict(list); identities=set(); seen_agents=defaultdict(set)
    for name in meta['names']:
        check_probe(name,run)
        a=read(run/'assignments'/f'{name}.json')
        for r in read(run/a['output']):
            identity=(r['unit_id'],r['question_id'],tuple(r['visible_document_ids']))
            if identity in identities:raise ValueError('Duplicate probe task')
            identities.add(identity)
            if r['agent'] in seen_agents[r['unit_id']]:raise ValueError('Agent saw multiple views/questions of same unit')
            seen_agents[r['unit_id']].add(r['agent'])
            mask='-'.join(r['visible_document_ids']) or 'none'
            groups[f'{mask}-{r["question_id"]}'].append(r)
    for name,records in groups.items():save_json(run/'reviews/probe'/f'{name}.json',sorted(records,key=lambda r:r['unit_id']))
    print(f'Merged {len(identities)} verified probes')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--prefix',default='independent-probe');p.add_argument('command',choices=['pack','merge'])
    a=p.parse_args();(pack if a.command=='pack' else merge)(a.run_dir.resolve(),a.prefix)

if __name__=='__main__':main()
