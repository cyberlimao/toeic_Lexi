"""Finish selected units of an existing plan with bounded, independent workers.

Uses the unchanged Engine gates. Each worker owns its SDK client and SQLite
connection; each unit has an OS lock. No model call contains multiple groups.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from toeic_engine.cli import export, create_provider
from toeic_engine.engine import Engine, prompt_version, save_json, verify_entry
from toeic_engine.novelty import History


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--end',type=int)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    p.add_argument('--max-calls-per-worker',type=int,default=70)
    p.add_argument('--max-attempts',type=int,default=3)
    p.add_argument('--no-export',action='store_true',help='Write unit checkpoints only when another coordinator owns bundle exports')
    args=p.parse_args()
    out=args.run.resolve()
    run=json.loads((out/'run.json').read_text())
    if run['prompt_version']!=prompt_version():
        raise ValueError('Prompts changed; use an explicit audit or new run')
    end=args.end or len(run['plans'])
    if not 1 <= args.start <= end <= len(run['plans']):
        raise ValueError('Invalid unit range')
    invocation=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    tasks=queue.Queue()
    for index in range(args.start-1,end): tasks.put(index)
    guard=threading.Lock()
    errors=[]
    counters={}
    (out/'locks').mkdir(exist_ok=True)
    def progress(message):
        with guard: print(message,flush=True)
    def snapshot():
        if args.no_export: return
        entries=[]
        for i,plan in enumerate(run['plans'],1):
            path=out/'units'/f'u{i:03d}.json'
            if path.exists():
                entry=json.loads(path.read_text())
                if entry['blueprint']!=plan or verify_entry(entry):
                    raise ValueError(f'Invalid checkpoint {path}')
                entries.append(entry)
        if entries: export(out,run,entries)
    def worker(worker_id):
        history=History(Path(run['history']))
        provider=create_provider(run.get('author_provider','gemini'),run.get('review_provider','gemini'),
                                 out/'api'/f'{invocation}-w{worker_id}',args.max_calls_per_worker,900)
        engine=Engine(provider,history,out,args.max_attempts,progress)
        try:
            while True:
                try: index=tasks.get_nowait()
                except queue.Empty: break
                uid=f'u{index+1:03d}'
                try:
                    with (out/'locks'/f'{uid}.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        checkpoint=out/'units'/f'{uid}.json'
                        pending=out/'units'/f'{uid}.pending.json'
                        if not checkpoint.exists() and pending.exists():
                            entry=json.loads(pending.read_text())
                            if entry['blueprint']==run['plans'][index] and not verify_entry(entry) and history.has_accepted(f'{out}/{uid}',entry['content_sha256']):
                                pending.replace(checkpoint)
                        if checkpoint.exists():
                            entry=json.loads(checkpoint.read_text())
                            if entry['blueprint']!=run['plans'][index] or verify_entry(entry):
                                raise ValueError(f'Invalid checkpoint {uid}')
                            progress(f'{uid} · using accepted checkpoint')
                        else:
                            engine.generate_unit(run['plans'][index],uid)
                        with guard: snapshot()
                except Exception as exc:
                    with guard:
                        errors.append({'unit_id':uid,'error':str(exc)})
                        print(f'{uid} · retained for repair: {exc}',flush=True)
                finally: tasks.task_done()
        finally:
            counters[worker_id]=provider.calls
            provider.close();history.close()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker,range(1,args.workers+1)))
    snapshot()
    save_json(out/'worker-runs'/f'{invocation}.json',{'workers':args.workers,'selected_units':[args.start,end],
        'max_calls_per_worker':args.max_calls_per_worker,'calls':counters,'failures':errors,
        'author_provider':run.get('author_provider','gemini'),'review_provider':run.get('review_provider','gemini'),
        'note':'Independent per-unit generation; same Engine quality gates and transaction-protected novelty history.'})
    print(f'Worker run complete: {sum(counters.values())} calls; {len(errors)} units require attention',flush=True)
    return bool(errors)


if __name__=='__main__':
    raise SystemExit(main())
