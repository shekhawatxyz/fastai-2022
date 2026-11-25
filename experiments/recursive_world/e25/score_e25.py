"""Strict official E25 cohort scorer. No missing cells, no duplicate episodes.
Use: python score_e25.py test/*.jsonl --upstream-logs /path/to/outputs/planning/replan
Without upstream logs, scores are provisional (baseline parity not verified).
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

DATASETS=('hand','cup','fetch','bluerov')
ENCODERS=('mae_cls','dinov2s_cls')
CONDITIONS=('persistent','fresh','none')
ARMS=('trained_predictor','counterfactual','scalar','bias','actuator','joint')


def score(paths, upstream=None, scope="full"):
    cells = [("hand", "mae_cls", 0)] if scope == "pilot" else list(itertools.product(DATASETS,ENCODERS,range(3)))
    expected=set((d,e,s,c,i,a) for d,e,s in cells for c,a in itertools.product(CONDITIONS,ARMS) for i in range(100*s,100*s+100))
    rows={}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r=json.loads(line)
            if r['mode']!='test' or r['seed_base']!=3109000000:
                raise ValueError('Not a frozen official-test record')
            k=tuple(r[v] for v in ('dataset','encoder','seed','condition','episode','arm'))
            if k not in expected or k in rows: raise ValueError(f'Unexpected or duplicated {k}')
            distances=np.asarray(r['distances'],float)
            if distances.shape!=(3,) or not np.all(np.isfinite(distances)) or abs(distances[-1]-r['final_distance'])>1e-9 or bool(r['success'])!=bool(distances[-1]<.2):
                raise ValueError(f'Invalid metric {k}')
            rows[k]=r
    if set(rows)!=expected: raise ValueError(f'Incomplete cohort: {len(rows)}/{len(expected)} rows; no gate score')
    # Bootstrap episodes within each dataset/encoder/seed cell. Preserve pairing across conditions and arms.
    x=np.array([[[[rows[d,e,s,c,i,a]['success'] for a in ARMS] for c in CONDITIONS] for i in range(100*s,100*s+100)] for d,e,s in cells],dtype=float)
    means=x.mean((0,1)); rng=np.random.default_rng(250025)
    contrasts=[]
    for _ in range(3000):
        sample=x[np.arange(len(cells))[:,None],rng.integers(100,size=(len(cells),100))].mean((0,1))
        contrasts.append((sample[:,[-1]]-sample[:,:-1]))
    b=np.asarray(contrasts); delta=means[:,[-1]]-means[:,:-1]; overall=delta.mean(0)
    # Simultaneous one-sided 95% lower bounds over all five overall comparisons.
    lower=np.quantile(b.mean(1),.05/5,axis=0)
    j=ARMS.index('counterfactual')
    parity=None
    if upstream:
        errors=[]; count=0
        for d,e,s in cells:
            f=Path(upstream)/f'{d}_{e}_s{s}'/'episodes.jsonl'
            found=set()
            for line in f.read_text().splitlines():
                r=json.loads(line)
                if r['task']!='replan' or r['arm'] not in ARMS[:2]:continue
                k=(d,e,s,r['condition'],r['episode'],r['arm'])
                if k not in rows or k in found: raise ValueError(f'Upstream mismatch {k}')
                found.add(k); count+=1
                errors.append(abs(r['final_distance']-rows[k]['final_distance']))
            if len(found)!=600:raise ValueError(f'Incomplete upstream parity {d,e,s}: {len(found)}')
        parity={'n':count,'max_absolute_distance_difference':max(errors),'pass':max(errors)<1e-5}
    # Original gate plus comparison to all implemented controls, not a SOTA claim.
    gate=(delta[2,j]>=.08 and delta[0,j]>=-.02 and delta[1,j]>=-.02 and overall[j]>=.03 and np.all(lower>0))
    return {'n_records':len(rows),'success_percent':{c:{a:100*means[i,j] for j,a in enumerate(ARMS)} for i,c in enumerate(CONDITIONS)},'joint_minus_controls_pp':dict(zip(ARMS[:-1],100*overall)),'simultaneous_lower_pp':dict(zip(ARMS[:-1],100*lower)),'upstream_parity':parity,'implemented_control_gate':bool(gate),'scope':scope,'confirmed_gate':bool(scope=='full' and gate and parity and parity['pass']),'note':'ACTUS comparison and current external SOTA verification still required; no SOTA claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('files',nargs='+');p.add_argument('--upstream-logs',required=True);p.add_argument('--scope',choices=['pilot','full'],default='full');a=p.parse_args()
    print(json.dumps(score(a.files,a.upstream_logs,a.scope),indent=2))
