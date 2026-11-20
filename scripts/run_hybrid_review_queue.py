"""Run explicitly enqueued native drafts through bounded parallel Claude review."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from toeic_engine.engine import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError('Use one to six bounded workers')
    run = args.run.resolve()
    jobs = run / 'jobs'
    jobs.mkdir(exist_ok=True)
    active = {}
    try:
        while True:
            for name, (process, handle, spec) in list(active.items()):
                code = process.poll()
                if code is not None:
                    handle.close()
                    save_json(jobs / f'{name}.result.json', {'job':spec, 'exit_code':code,
                        'status':'accepted_automated' if code == 0 else 'needs_attention'})
                    print(f'{name}: exit {code}', flush=True)
                    del active[name]
            stopping = (run / 'STOP_QUEUE').exists()
            if not stopping:
                for path in sorted(jobs.glob('*.job.json')):
                    name = path.name.removesuffix('.job.json')
                    if len(active) >= args.workers:
                        break
                    if name in active or (jobs / f'{name}.result.json').exists():
                        continue
                    spec = json.loads(path.read_text())
                    cmd = [sys.executable, '-u', '-m', 'scripts.hybrid_reviewed_test', 'review-unit',
                           '--run',str(run),'--unit',spec['unit_id'],'--draft',spec['draft'],
                           '--author-agent',spec['author_agent']]
                    handle = (jobs / f'{name}.log').open('a')
                    process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
                    active[name] = (process, handle, spec)
                    print(f'{name}: started', flush=True)
            if stopping and not active:
                break
            time.sleep(3)
    finally:
        for process, handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
            handle.close()


if __name__ == '__main__':
    main()
