"""Run pinned CWM replan cells, then score; no silent skipping or fallback."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIN = '5c84bba92b2ff19d41175070a61238fa1396d10d'
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--scope', choices=['pilot', 'full'], required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    up, work = args.upstream.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CWM_OUTPUTS=str(work/'outputs'), MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl', SDL_VIDEODRIVER='dummy', WANDB_MODE='disabled')
    def run(*cmd, capture=False):
        cmd = list(map(str, cmd))
        print('>', ' '.join(cmd), flush=True)
        return subprocess.run(cmd, cwd=up, env=env, check=True, text=True,
                              stdout=subprocess.PIPE if capture else None).stdout
    def py(*cmd, capture=False):
        return run(sys.executable, *cmd, capture=capture)
    if run('git', 'rev-parse', 'HEAD', capture=True).strip() != PIN:
        raise RuntimeError('Wrong upstream revision')
    py(HERE/'test_e25.py')
    cells = [('hand', 'mae_cls', 0)] if args.scope == 'pilot' else list(itertools.product(
        ('hand', 'cup', 'fetch', 'bluerov'), ('mae_cls', 'dinov2s_cls'), range(3)))
    def index(script, tag, key):
        return py(script, tag, '--list', capture=True).splitlines().index(key)
    def ready(path, n):
        if not path.exists(): return False
        try:
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            if len(rows) == n: return True
        except ValueError: pass
        path.rename(path.with_suffix(f'.partial.{time.time_ns()}'))
        return False
    files = []
    for d,e,s in cells:
        fitd = 'bluerov32k' if d == 'bluerov' else d
        py('experiments/1_checkpoints/download.py', e, *(['cup'] if d == 'cup' else []))
        for dataset in dict.fromkeys((d, fitd)):
            py('experiments/2_data/collect.py', dataset)
            py('experiments/2_data/encode.py', dataset, e, '--device', args.device)
        py('experiments/2_data/population.py', d)
        py('experiments/3_train/readouts.py', d, e, '--index', 0, '--device', args.device)
        script = 'experiments/3_train/predictors.py'
        py(script, 'planning', '--index', index(script,'planning',f'{fitd}_{e}_h1_s{s}'), '--device', args.device)
        script = 'experiments/4_evaluate/planning.py'
        py(script, 'replan', '--index', index(script,'replan',f'{d} {e} s{s}'), '--device', args.device)
        name = f'{d}_{e}_s{s}.jsonl'
        cal, test = work/'calibration'/name, work/'test'/name
        base = [HERE/'run_e25.py','--upstream',up,'--dataset',d,'--encoder',e,'--seed',s,'--device',args.device]
        if not ready(cal,300): py(*base,'--out',cal)
        if not ready(test,1800): py(*base,'--calibration',cal,'--out',test)
        files.append(test)
    py(HERE/'score_e25.py', *files, '--scope', args.scope, '--upstream-logs', work/'outputs/planning/replan')

if __name__ == '__main__': main()
