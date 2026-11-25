"""Run the upstream closed-loop benchmark with six correction arms.
Requires the original CWM repository with generated model fits, populations and dependencies.
Calibration uses disjoint seed_base and trained_predictor only. Test condition is used
by the environment, never supplied to the correction. No GitHub execution.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from e25 import Correction, PlannerView, calibrate, ARMS


def run_episode(up, sim, m, design, idx, sigma, episode, condition, arm, noise, alpha):
    base, steps = design['seed_base'], design['planning_steps']
    if condition == 'none':
        drivers, goal_driver = np.zeros(steps+2), 0.
    elif condition == 'persistent':
        g = np.random.default_rng(np.random.SeedSequence([base,episode,11])).standard_normal()
        drivers, goal_driver = np.full(steps+2,g),g
    else:
        drivers = np.random.default_rng(np.random.SeedSequence([base,episode,12])).standard_normal(steps+2)
        goal_driver=0.
    seed=base+episode
    limit=sim.info['action_limit']
    goal_command=sim('fix',np.random.default_rng(np.random.SeedSequence([base,episode,14])).uniform(-limit,limit,m.a))
    sim('reset',seed); sim('behavior_rng',[base,episode,13]); warm=[]
    for t in range(2):
        warm.append(sim('behavior',t)); sim('step',warm[-1],drivers[t])
    goal=np.asarray(sim('step',goal_command,goal_driver))[idx]
    sim('reset',seed)
    tensor=lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32,device=m.device)
    c=Correction(m,sigma,limit,noise,alpha,arm)
    # Match upstream's render chronology exactly: only the second warm-up
    # transition is observed by the model at initialization.
    sim('step',warm[0],drivers[0]);zprev=m.encode(sim('render'))
    sim('step',warm[1],drivers[1]);z=m.encode(sim('render'))
    c.observe(zprev,tensor(warm[1]),m.decode(z))
    view=PlannerView(c)
    log=dict(episode=episode,condition=condition,arm=arm,distances=[],commands=[],observed_residuals=[])
    for t in range(steps):
        log['observed_residuals'].append(c.raw_last.cpu().tolist())
        generator=torch.Generator(device=m.device).manual_seed(int(base+7919*episode+t))
        a=up.cem(view,z,torch.zeros_like(c.sigma),tensor(goal),tensor(sigma),limit,generator,
                 design['cem_population'],design['cem_elites'],design['cem_iterations'])
        y=np.asarray(sim('step',a.cpu().numpy().astype(np.float64),drivers[t+2]))[idx]
        if t+1 < steps:
            znew=m.encode(sim('render')); c.observe(z,a,m.decode(znew));z=znew
        log['distances'].append(float(np.sqrt(np.mean(((y-goal)/sigma)**2))))
        log['commands'].append(a.cpu().tolist())
    log['final_distance']=log['distances'][-1]
    log['success']=log['final_distance']<design['success_tolerance']
    return log


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--dataset',choices=['hand','cup','fetch','bluerov'],required=True)
    p.add_argument('--encoder',choices=['mae_cls','dinov2s_cls'],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibration',type=Path,help='Disjoint calibration JSONL; omit to generate calibration only')
    p.add_argument('--count',type=int,default=100,help='Official count=100; other values smoke only')
    args=p.parse_args();root=args.upstream.resolve();sys.path.insert(0,str(root))
    if not 1 <= args.count <= 100: p.error('--count must be in 1..100')
    from cwm import registry
    from cwm.model import WorldModel
    from cwm.sims.worker import Remote
    spec=importlib.util.spec_from_file_location('upstream_planning',root/'experiments/4_evaluate/planning.py')
    up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
    design=json.loads((root/'experiments/configs/planning/replan.json').read_text())
    original_base=design['seed_base'];d=registry.dataset(args.dataset);cell=design['cells'][args.dataset]
    fit=registry.OUT/'fits'/registry.fit_key(dict(dataset=cell['fit'],encoder=args.encoder,history=1,seed=args.seed,recipe='final'))/'fit.pt'
    readout=registry.readout(d['readouts'],args.encoder,'state')
    for path in (fit,readout,registry.data(d['population'],'population','branches.npz')):
        if not path.is_file():raise FileNotFoundError(f'Missing upstream artifact: {path}')
    sigma=up.sigma_of(args.dataset)
    m=WorldModel(args.encoder,fit,readout,args.device)
    tag=dict(dataset=args.dataset,encoder=args.encoder,seed=args.seed)
    if args.calibration:
        records=[json.loads(line) for line in args.calibration.read_text().splitlines() if line.strip()]
        expected=dict(tag,seed_base=original_base+1_000_000,mode='calibration')
        if not records or any(any(r.get(k)!=v for k,v in expected.items()) for r in records):
            raise ValueError('Wrong calibration cell, mode, or seed base')
        keys=[(r['episode'],r['condition'],r['arm']) for r in records]
        wanted={(i,c,'trained_predictor') for i in range(100*args.seed,100*args.seed+args.count) for c in design['conditions']}
        if set(keys)!=wanted or len(keys)!=len(wanted): raise ValueError('Incomplete or duplicated calibration cohort')
        noise,alpha=calibrate(records,sigma);arms=ARMS;mode='test'
    else:
        design['seed_base']+=1_000_000;noise=np.ones_like(sigma);alpha=0.;arms=['trained_predictor'];mode='calibration'
    sim=Remote(registry.simulator(args.dataset,'population'),up.sim_command(args.dataset))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    try:
        with args.out.open('x') as f: # no overwrite or accidental duplicate episodes
            for episode in range(100*args.seed,100*args.seed+args.count):
                for condition in design['conditions']:
                    for arm in arms:
                        r=run_episode(up,sim,m,design,d['coordinates'],sigma,episode,condition,arm,noise,alpha)
                        r.update(tag,seed_base=design['seed_base'],mode=mode,noise=list(map(float,noise)),alpha=alpha)
                        f.write(json.dumps(r)+'\n');f.flush()
    finally: sim.close()

if __name__=='__main__':main()
