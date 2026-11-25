"""Recursive rule induction for action-conditioned world models.

Run on Colab: python recursive_world.py --suite synthetic --device cuda
For ARC: --suite arc --warc-root /content/WARC (see README in this directory).
The MechEval paper numbers are references, NOT scores on our replay proxy.
No game source, mechanic labels, or future observations enter prediction.
"""
import argparse
from dataclasses import dataclass
from functools import lru_cache
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

C, K = 16, 12
OFF = ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1),
       (-2,0),(2,0),(0,-2),(0,2))
O = len(OFF) + C
# Reference only: https://github.com/third-intelligence/WARC#evaluation
REFERENCE = {
    "episodic": (32.9,109,76.8), "WARC frozen": (18.4,114,85.5),
    "WARC scratch": (35.0,105,51.2), "WARC GRU": (24.0,87,84.1),
    "DreamerV3": (25.9,114,90.8), "WARC TTT": (69.1,74,18.8),
}
FOLDS = {1:"bp35 lf52 lp85 r11l re86 sb26 sc25 su15 wa30".split(),
         2:"cd82 cn04 dc22 m0r0 sk48 tn36 tr87 tu93".split(),
         3:"ar25 ft09 g50t ka59 ls20 s5i5 sp80 vc33".split()}
ARMS = ("direct", "copy", "static", "recursive", "fixed", "small", "stale", "grow")


@dataclass
class Sample:
    h: np.ndarray             # three frames, 16 = missing history
    a: np.ndarray             # two past actions + action to predict
    y: np.ndarray
    reset: bool = False

    @property
    def s(self): return self.h[-1]


def pack(samples, device):
    return tuple(torch.as_tensor(np.stack([getattr(s,k) for s in samples]),
                                 device=device, dtype=torch.long) for k in ("h","a","y"))


@lru_cache(maxsize=16)
def source_indices(h,w,device):
    """Every copy primitive is a relative address. Outside the board means KEEP."""
    y,x = torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing="ij")
    return torch.stack([torch.where((y+dy>=0)&(y+dy<h)&(x+dx>=0)&(x+dx<w),
                       (y+dy)*w+x+dx,y*w+x).flatten() for dy,dx in OFF])


def onehot(x,n=C):
    return F.one_hot(x,n).movedim(-1,-3).float()


def sources(board):
    """Gather every relative copy in one operation; no Python loop over cells."""
    b,c,h,w=board.shape
    idx=source_indices(h,w,board.device)
    return board.flatten(2).index_select(2,idx.flatten()).reshape(b,c,len(OFF),h*w)


def actions(a,h,w):
    ids=F.one_hot(a[...,0],7).float().reshape(len(a),21,1,1).expand(-1,-1,h,w)
    yy=torch.arange(h,device=a.device)[None,None,:,None]
    xx=torch.arange(w,device=a.device)[None,None,None,:]
    click=((a[...,0]==6)[...,None,None] & (xx==a[...,1,None,None]) &
           (yy==a[...,2,None,None])).float()
    return torch.cat((ids,click),1)


class World(nn.Module):
    """Labels are executable effect slots, not target colors or fixed mechanic IDs.

    Each pass reads the original encoding, current canvas and previous label field.
    It selects a label at each position; that label executes a copy/write operation.
    The next pass can apply an operation to the result of the previous operation.
    Slot meaning and applicability are independently trainable.
    """
    def __init__(self,arm,width=32,passes=3):
        super().__init__()
        self.arm,self.passes=arm,passes
        self.symbolic=arm not in ("direct","copy")
        self.dynamic=arm in ("grow","stale")
        n=C if arm=="direct" else O if arm=="copy" else K
        self.enc=nn.Conv2d(3*(C+1)+24,width,3,padding=1)
        self.refine=nn.Sequential(nn.Conv2d(2*width+C+K,width,3,padding=1),
            nn.GroupNorm(4,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),
            nn.GroupNorm(4,width),nn.SiLU())
        self.guard=nn.Conv2d(width,n,1)
        self.feedback=nn.Conv2d(n,K,1)
        self.register_buffer("active",torch.arange(K)<(4 if self.dynamic or arm=="small" else K))
        self.effects=nn.Parameter(torch.randn(K,O)*.1) if self.symbolic else None
        if self.symbolic:
            with torch.no_grad():
                anchors=torch.cat((torch.zeros(1,dtype=torch.long),torch.randperm(O-1)[:K-1]+1))
                self.effects[torch.arange(K),anchors]+=2
        with torch.no_grad():
            self.guard.bias.zero_()
            if arm!="direct": self.guard.bias[0]=1

    def forward(self,h,a,hard=False,details=False):
        original=onehot(h[:,-1]); board=original
        base=self.enc(torch.cat((onehot(h,C+1).flatten(1,2),actions(a,*h.shape[-2:])),1))
        hidden=base
        label=base.new_zeros((len(h),K,*h.shape[-2:]))
        table=self.effects.float().softmax(-1) if self.symbolic else None
        if hard and self.symbolic: table=F.one_hot(table.argmax(-1),O).float()
        for _ in range(self.passes):
            # Static control receives identical recurrent compute but cannot execute on
            # the newly produced canvas or reuse the previous label field.
            canvas=original if self.arm=="static" else board
            prior=torch.zeros_like(label) if self.arm=="static" else label
            hidden=hidden+self.refine(torch.cat((base,hidden,canvas,prior),1))
            logits=self.guard(hidden).float()
            if self.symbolic: logits=logits.masked_fill(~self.active[None,:,None,None],-1e9)
            q=logits.softmax(1)
            if hard: q=F.one_hot(q.argmax(1),q.shape[1]).movedim(-1,1).float()
            label=self.feedback(q)
            if self.arm=="direct": board=q; continue
            # B x O x H x W. Use FP32 for the probability/execution algebra.
            with torch.autocast(h.device.type,enabled=False):
                w=q if table is None else torch.einsum("bkhw,ko->bohw",q,table)
                b,_,hh,ww=w.shape
                copy=torch.einsum("bcon,bon->bcn",sources(canvas),w[:,:len(OFF)].flatten(2))
                board=copy.reshape(b,C,hh,ww)+w[:,len(OFF):]
                board=board/board.sum(1,keepdim=True).clamp_min(1e-8)
        return (board,q,table,canvas) if details else board

    @torch.no_grad()
    def split(self,parent,optimizer):
        """Birth of a label: duplicate an existing operation, then let evidence
        differentiate its applicability and semantics. No mechanic is hardcoded.
        Reserve capacity avoids rebuilding the optimizer on every birth.
        """
        free=(~self.active).nonzero().flatten()
        if not len(free): return None
        child=int(free[0])
        self.guard.weight[child].copy_(self.guard.weight[parent])
        self.guard.weight[child].add_(torch.randn_like(self.guard.weight[child])*.01)
        self.guard.bias[parent]-=math.log(2)
        self.guard.bias[child].copy_(self.guard.bias[parent])
        self.effects[child].copy_(self.effects[parent])
        self.feedback.weight[:,child].copy_(self.feedback.weight[:,parent])
        # Clear changed coordinates' Adam moments, not unrelated parameters.
        for param,dim in ((self.guard.weight,0),(self.guard.bias,0),(self.effects,0),(self.feedback.weight,1)):
            for value in optimizer.state.get(param,{}).values():
                if torch.is_tensor(value) and value.shape==param.shape:
                    value.select(dim,parent).zero_(); value.select(dim,child).zero_()
        self.active[child]=True
        return child


@torch.no_grad()
def relabel(detail,y):
    """Given a known outcome, infer which executable labels explain it.
    Re-evaluate old transitions under today's dictionary; never use this at prediction.
    Soft responsibilities preserve ambiguity rather than inventing a unique explanation.
    """
    _,q,table,canvas=detail
    b,c,h,w=canvas.shape
    target=onehot(y)
    with torch.autocast(y.device.type,enabled=False):
        matches=(sources(canvas)*target.flatten(2)[:,:,None,:]).sum(1).reshape(b,len(OFF),h,w)
        likelihood=torch.einsum("ko,bohw->bkhw",table,torch.cat((matches,target),1))
        posterior=q*likelihood
        return posterior/posterior.sum(1,keepdim=True).clamp_min(1e-9)


def balanced(x,changed):
    # Do not let thousands of unchanged pixels bury the learning signal.
    return .5*(x*changed).sum()/changed.sum().clamp_min(1)+.5*(x*~changed).sum()/(~changed).sum().clamp_min(1)


class Learner:
    def __init__(self,arm,device,seed,lr=0.0003):
        torch.manual_seed(seed)
        self.device,self.arm=torch.device(device),arm
        self.model=World(arm).to(device)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=.01,
                                   fused=self.device.type=="cuda")
        self.rng=np.random.default_rng(seed)
        self.data=[]; self.groups=[]; self.labels={}; self.updates=0
        self.amp=self.device.type=="cuda" and torch.cuda.is_bf16_supported()
        self.initial=self.snapshot(); self.previous=self.initial
        self.events=[]; self.revision_distance=0.; self.revision_batches=0; self.loss=0.; self.grad=0.

    def snapshot(self):
        return {n:p.detach().float().cpu().clone() for n,p in self.model.named_parameters()}

    def drift(self,epoch):
        now=self.snapshot()
        rows=[dict(epoch=epoch,parameter=n,
                   relative_init=float((v-self.initial[n]).norm()/self.initial[n].norm().clamp_min(1e-8)),
                   rms_step=float((v-self.previous[n]).square().mean().sqrt()),
                   cosine=float(F.cosine_similarity(v.flatten(),self.initial[n].flatten(),dim=0)))
              for n,v in now.items()]
        self.previous=now
        return rows

    @torch.no_grad()
    def predict(self,s,hard=False):
        self.model.eval()
        h,a,_=pack([s],self.device)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.amp and not hard):
            return self.model(h,a,hard=hard)[0].float().cpu().numpy()

    @torch.no_grad()
    def observe(self,views):
        # Each replay item is a group of symmetry views of one observed transition.
        self.groups.append(list(range(len(self.data),len(self.data)+len(views))))
        for view in views:
            h,a,y=pack([view],self.device)
            i=len(self.data); self.data.append((h[0],a[0],y[0]))
            if self.arm=="stale":
                self.labels[i]=relabel(self.model(h,a,details=True),y)[0].half()

    def batch(self,ids):
        return tuple(torch.stack(xs) for xs in zip(*(self.data[int(i)] for i in ids)))

    def step(self,ids):
        h,a,y=self.batch(ids)
        changed=y!=h[:,-1]
        self.model.train(); self.opt.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.amp):
            detail=self.model(h,a,details=True); pred,q,_,_=detail
            loss=balanced(-pred.gather(1,y[:,None]).squeeze(1).clamp_min(1e-7).log(),changed)
            if self.arm in ("fixed","small","grow","stale"):
                fresh=relabel(detail,y)
                if self.arm=="stale":
                    target=torch.stack([self.labels[int(i)] for i in ids]).float()
                else:
                    target=fresh
                    # Measure, rather than assume, that old interpretations change.
                    old=[(j,self.labels[int(i)]) for j,i in enumerate(ids) if int(i) in self.labels]
                    if old:
                        js,qs=zip(*old)
                        self.revision_distance+=float((fresh[list(js)]-torch.stack(qs).float()).abs().sum(1).mean().detach())
                        self.revision_batches+=1
                    for i,v in zip(ids,fresh): self.labels[int(i)]=v.half()
                loss=loss+.2*balanced(-(target*q.clamp_min(1e-7).log()).sum(1),changed)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
        loss.backward()
        norm=nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        self.opt.step(); self.updates+=1
        self.loss,self.grad=float(loss.detach()),float(norm)

    @torch.no_grad()
    def reconsider(self,epoch):
        """Split only on observed residuals; no evaluation set drives growth."""
        if not self.model.dynamic or self.model.active.all() or epoch%4: return
        # Use the same recent replay for both dynamic ablations.
        ids=[g[0] for g in self.groups[-8:]]
        h,a,y=self.batch(ids)
        pred,q,_,_=self.model(h,a,details=True)
        mask=(y!=h[:,-1])
        if not mask.any(): return
        error=(pred.argmax(1)!=y)&mask
        if float(error.sum()/mask.sum())<.15: return
        parent=int((q*error[:,None]).sum((0,2,3)).argmax())
        child=self.model.split(parent,self.opt)
        if child is not None: self.events.append(dict(epoch=epoch,parent=parent,child=child))

    def offline_epoch(self,epoch,batch=8):
        for ids in np.array_split(self.rng.permutation(len(self.data)),math.ceil(len(self.data)/batch)):
            self.step(ids)
        self.reconsider(epoch)

    def online(self,epoch,steps=20,batch=8):
        # WARC's newest 10 + 20 older transitions. Each group includes rotations
        # for ARC; sample uniformly across the resulting views.
        n=len(self.groups); cut=max(0,n-10)
        old=self.rng.choice(cut,min(cut,20),replace=False)
        bank=np.array([i for g in [*old,*range(cut,n)] for i in self.groups[g]])
        for _ in range(steps): self.step(self.rng.choice(bank,min(batch,len(bank)),replace=False))
        self.reconsider(epoch)

    def vocabulary(self,epoch):
        if not self.model.symbolic: return []
        t=self.model.effects.detach().softmax(-1).cpu().numpy()
        return [dict(epoch=epoch,slot=k,active=bool(self.model.active[k]),
                     primitive=int(v.argmax()),confidence=float(v.max()),
                     entropy=float(-(v*np.log(v+1e-9)).sum())) for k,v in enumerate(t)]


def metrics(s,y,p):
    ch=s!=y; edit=s!=p; ok=p==y
    return dict(changed_acc=float(ok[ch].mean()) if ch.any() else np.nan,
                false_edit=float(edit[~ch].mean()) if (~ch).any() else np.nan,
                exact=float(ok.all()),cell_acc=float(ok.mean()),
                change_f1=2*int((edit&ch).sum())/max(1,int(edit.sum()+ch.sum())))


def synthetic(seed,n,ood=False,task="move"):
    """Sanity checks, not a claim about ARC. Includes a non-copy task."""
    rng=np.random.default_rng(seed); out=[]
    for _ in range(n):
        s=np.zeros((12,12),np.uint8)
        for _ in range(15):
            r,c=rng.integers(1,11,2); s[r,c]=rng.choice([5,6,7] if ood else [1,2,3])
        for _ in range(12):
            r,c=rng.integers(0,12,2); s[r,c]=9
        a=int(rng.integers(1,5)); dy,dx=OFF[a]; y=s.copy()
        if task=="literal": y[s!=0]=(s[s!=0]+a)%C
        else:
            for _ in range(2 if task=="cascade" else 1):
                z=y.copy()
                for r,c in zip(*np.where((y!=0)&(y!=9))):
                    u,v=r+dy,c+dx
                    if 0<=u<12 and 0<=v<12 and y[u,v]==0: z[r,c]=0; z[u,v]=y[r,c]
                y=z
        h=np.stack((np.full_like(s,16),np.full_like(s,16),s))
        out.append(Sample(h,np.array([[0,-1,-1],[0,-1,-1],[a,-1,-1]]),y,True))
    return out


def parse(records,limit):
    """Convert recorded observations into causal windows; exclude reset and level transitions."""
    frames=[]; inter=[]; level=None; out=[]; reset=True
    for event in records:
        d=event.get("data",event)
        if not isinstance(d,dict): continue
        try:
            f=np.asarray(d["frame"])
            if f.ndim==3: f=f[-1]
            if f.shape!=(64,64) or not np.issubdtype(f.dtype,np.number) or np.any((f<0)|(f>=C)): raise ValueError()
            f=f.astype(np.uint8)
            ai=d.get("action_input") or {}; ad=ai.get("data") or {}; a=int(ai.get("id",0))
            act=(a,int(ad["x"]) if a==6 else -1,int(ad["y"]) if a==6 else -1)
            lv=d.get("levels_completed",0)
        except (ValueError,TypeError,KeyError,IndexError):
            frames=[]; inter=[]; level=None; reset=True; continue
        valid=bool(frames) and level==lv and 1<=a<=6 and not d.get("full_reset",False) and str(d.get("state","")).upper() not in ("WIN","GAME_OVER")
        if valid:
            h=np.stack([np.full_like(f,16)]*(3-len(frames))+frames)
            acts=np.array([(0,-1,-1)]*(2-len(inter))+inter+[act])
            out.append(Sample(h,acts,f,reset)); reset=False
            frames=(frames+[f])[-3:]; inter=(inter+[act])[-2:]
            if len(out)>=limit: break
        else: frames=[f]; inter=[]; reset=True
        level=lv
    return out


def arc_streams(args):
    counts={g:0 for g in args.games.split(",")}
    if args.data:
        def records():
            for path in sorted(Path(args.data).rglob("*.recording.jsonl")):
                g=next((g for g in counts if any(p.split("-")[0]==g for p in path.parts) or path.name.startswith(g)),None)
                if g: yield g,str(path),[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    else:
        from datasets import load_dataset
        def records():
            for row in load_dataset("magic-sword/arc_agi_3_public_demo_human_testing",split="train",streaming=True):
                g=str(row["env"]).lower().split("-")[0]
                if g not in counts: continue
                raw=row["trajectory"]
                if isinstance(raw,str):
                    try: raw=json.loads(raw)
                    except json.JSONDecodeError: raw=[json.loads(x) for x in raw.splitlines() if x.strip()]
                if isinstance(raw,dict): raw=raw.get("trajectory",raw.get("records",raw.get("events",[])))
                yield g,str(row["guid"]),raw
    for g,sid,raw in records():
        if counts[g]>=args.sessions: continue
        s=parse(raw,args.limit)
        if len(s)<min(12,args.limit): continue
        counts[g]+=1; yield g,sid,s
        if all(v>=args.sessions for v in counts.values()): return
    if any(v<args.sessions for v in counts.values()): raise RuntimeError(f"Insufficient recordings: {counts}")


def rotation_views(s):
    # Use the baseline's actual geometric/action convention, not guessed key IDs.
    from agent_base.src.augment import augment_one,GEOMETRIC_TRANSFORMS_NO_FLIP
    out=[s]
    for geo in GEOMETRIC_TRANSFORMS_NO_FLIP:
        if geo=="identity": continue
        h=[]; a=[]
        for frame,act in zip(s.h,s.a):
            x,aid,cx,cy,y=augment_one((frame,*act,s.y),geo,None)
            h.append(x); a.append((aid,cx,cy))
        out.append(Sample(np.stack(h),np.array(a),y,s.reset))
    return out


def warc_scores(game,stream,arm,seed,device):
    from huggingface_hub import hf_hub_download
    from warc_model.final_ttt import FinalPredictor
    from hist_window.src.history import HistoryWindow
    fold=next(f for f,gs in FOLDS.items() if game in gs)
    ckpt=hf_hub_download("third-intelligence/WARC",f"fold{fold}/checkpoint_best.pt")
    torch.manual_seed(seed)
    if arm=="scratch":
        # Same released architecture and online recipe, without pretrained parameters.
        # Temporarily substitute its constructor; let FinalPredictor initialize its own optimizers.
        from warc_model import final_ttt as ft
        from warc_model.model import build_model,UNIFIED_DEFAULTS
        original=ft.build_from_checkpoint
        def scratch(path,*,device):
            ck=torch.load(path,map_location="cpu",weights_only=False)
            mc=ck["config"]["model_config"]
            model,_=build_model(hist_n=mc["hist_n"],patch_size=mc["patch_size"],device=device,
                                **{k:mc[k] for k in UNIFIED_DEFAULTS if k in mc})
            return model,{k:v for k,v in ck.items() if k!="model_state"}
        ft.build_from_checkpoint=scratch
        try: predictor=FinalPredictor(ckpt,game=game,arm="ttt",device=device,seed=seed,strict_split=True)
        finally: ft.build_from_checkpoint=original
    else:
        predictor=FinalPredictor(ckpt,game=game,arm=arm,device=device,seed=seed,strict_split=True)
    window=HistoryWindow(n=3)
    for i,s in enumerate(stream):
        if s.reset or not len(window): window.reset(s.s)
        assert np.array_equal(window.current,s.s)
        a=tuple(map(int,s.a[-1]))
        started=time.perf_counter(); _,p=predictor.predict(window,[a])
        torch.cuda.synchronize(); pt=time.perf_counter()-started
        row=metrics(s.s,s.y,np.asarray(p[0]))
        started=time.perf_counter(); predictor.observe(window,a,s.y); window.push(a,s.y)
        torch.cuda.synchronize()
        yield dict(step=i,arm="WARC "+arm,seed=seed,predict_s=pt,update_s=time.perf_counter()-started,**row)
    del predictor; gc.collect(); torch.cuda.empty_cache()


def profile_update(device,out,samples=None):
    """One separate warmed-up training step: profiler overhead cannot affect scores."""
    from torch.profiler import profile,ProfilerActivity,record_function
    l=Learner("grow",device,42)
    for s in (samples or synthetic(42,8))[:8]: l.observe([s])
    ids=list(range(len(l.data)))
    l.step(ids)
    activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device=="cuda" else [])
    with profile(activities=activities,record_shapes=True,profile_memory=True) as p:
        with record_function("recursive_update"):
            l.step(ids)
            if device=="cuda": torch.cuda.synchronize()
    p.export_chrome_trace(str(out/"trace.json"))
    table=p.key_averages().table(sort_by="self_cuda_time_total" if device=="cuda" else "self_cpu_time_total",row_limit=15)
    (out/"profile.txt").write_text(table); print(table)


def selftest(device):
    s=synthetic(0,1)[0]
    for arm in ARMS:
        l=Learner(arm,device,0)
        p=l.predict(s)
        assert p.shape==(C,12,12) and np.isfinite(p).all() and np.allclose(p.sum(0),1,atol=1e-4)
        # No label/outcome leakage from a query.
        assert np.array_equal(p,l.predict(Sample(s.h,s.a,np.zeros_like(s.y))))
        if l.model.symbolic:
            assert np.isin(l.predict(s,True),[0,1]).all()
        l.observe([s]); l.step([0])
        assert any(r["rms_step"]>0 for r in l.drift(1))
        if l.model.dynamic:
            n=int(l.model.active.sum()); l.model.split(0,l.opt); assert int(l.model.active.sum())==n+1
    print("Self-tests passed",flush=True)


def run(args):
    import pandas as pd
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    scores=[]; drift=[]; logs=[]; vocab=[]; births=[]; profile_samples=None
    arms=args.arms.split(","); assert all(a in ARMS for a in arms)
    def trace(l,epoch,key):
        logs.append(dict(**key,epoch=epoch,loss=l.loss,grad_norm=l.grad,updates=l.updates,
                         revision_distance=l.revision_distance/max(1,l.revision_batches),active=int(l.model.active.sum()),
                         params=sum(p.numel() for p in l.model.parameters())))
        drift.extend(dict(**key,**r) for r in l.drift(epoch))
        vocab.extend(dict(**key,**r) for r in l.vocabulary(epoch))
        l.revision_distance=0.; l.revision_batches=0
    def save():
        for name,rows in (("scores",scores),("epochs",logs),("weight_drift",drift),("vocabulary",vocab),("births",births)):
            pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False)
    def sync():
        if args.device=="cuda": torch.cuda.synchronize()
    if args.suite in ("smoke","synthetic"):
        tasks=("move",) if args.suite=="smoke" else ("move","cascade","literal")
        for task in tasks:
            test=synthetic(99999,24,True,task)
            for arm in arms:
                members=[]
                for seed in range(args.seeds):
                    l=Learner(arm,args.device,seed,lr=.002)
                    for s in synthetic(seed,48 if args.suite!="smoke" else 8,False,task): l.observe([s])
                    key=dict(task=task,arm=arm,seed=seed)
                    for ep in range(1,args.epochs+1):
                        sync(); t=time.perf_counter(); l.offline_epoch(ep); sync()
                        trace(l,ep,key); logs[-1]["seconds"]=time.perf_counter()-t
                    probs=np.stack([l.predict(s) for s in test]); members.append(probs)
                    for i,(s,p) in enumerate(zip(test,probs)):
                        scores.append(dict(**key,step=i,**metrics(s.s,s.y,p.argmax(0))))
                        if l.model.symbolic:
                            scores.append(dict(task=task,arm=arm+"_hard",seed=seed,step=i,**metrics(s.s,s.y,l.predict(s,True).argmax(0))))
                    births.extend(dict(**key,**r) for r in l.events); save()
                # All members see the same test. Ensemble is explicitly extra compute.
                if args.ensemble and len(members)>1:
                    for i,(s,p) in enumerate(zip(test,np.mean(members,axis=0))):
                        scores.append(dict(task=task,arm=arm+"_ensemble",seed=-1,step=i,**metrics(s.s,s.y,p.argmax(0))))
    else:
        root=Path(args.warc_root).resolve()
        if not (root/"agent"/"warc_model").exists(): raise FileNotFoundError(root)
        sys.path[:0]=[str(root/"agent"),str(root/"agent"/"VARC")]
        if args.device!="cuda": raise ValueError("ARC WARC comparison requires CUDA")
        if not torch.cuda.is_bf16_supported():
            # T4 has no native BF16. Keep upstream code unchanged on disk; change
            # only the context manager used by the released predictor at runtime.
            os.environ["WARC_PREDICT_BF16"]="0"
            from warc_model import final_ttt as ft
            class FP32Torch:
                def __getattr__(self,name): return getattr(torch,name)
                def autocast(self,*a,**kw):
                    kw["enabled"]=False
                    return torch.autocast(*a,**kw)
            ft.torch=FP32Torch()
            print("WARC BF16 unsupported; both models use FP32. Not identical to published precision.")
        for game,sid,stream in arc_streams(args):
            if profile_samples is None: profile_samples=stream[:8]
            key=dict(game=game,session=sid)
            # Simple no-training controls on the same stream; never use future examples.
            history=[]
            for i,s in enumerate(stream):
                scores.append(dict(**key,arm="keep",seed=0,step=i,**metrics(s.s,s.y,s.s)))
                candidates=[x for x in history if np.array_equal(x.a[-1],s.a[-1])]
                if candidates:
                    near=min(candidates,key=lambda x:np.count_nonzero(x.s!=s.s))
                    p=np.where(near.s!=near.y,near.y,s.s)
                else: p=s.s
                scores.append(dict(**key,arm="local retrieval",seed=0,step=i,**metrics(s.s,s.y,p)))
                history.append(s)
            for arm in arms:
                members=[]
                for seed in range(args.seeds):
                    l=Learner(arm,args.device,seed)
                    probabilities=[]
                    for i,s in enumerate(stream):
                        sync(); t=time.perf_counter(); probs=l.predict(s); p=probs.argmax(0); sync(); pt=time.perf_counter()-t
                        if args.ensemble: probabilities.append(probs)
                        row=metrics(s.s,s.y,p)
                        if l.model.symbolic and args.hard_arc:
                            hard=l.predict(s,True).argmax(0)
                            scores.append(dict(**key,arm=arm+"_hard",seed=seed,step=i,**metrics(s.s,s.y,hard)))
                        t=time.perf_counter(); l.observe(rotation_views(s)); l.online(i+1,args.steps); sync()
                        scores.append(dict(**key,arm=arm,seed=seed,step=i,predict_s=pt,update_s=time.perf_counter()-t,**row))
                        if (i+1)%8==0 or i==len(stream)-1: trace(l,i+1,dict(**key,arm=arm,seed=seed))
                    births.extend(dict(**key,arm=arm,seed=seed,**r) for r in l.events); save()
                    if args.ensemble: members.append(np.stack(probabilities))
                    del l; gc.collect(); torch.cuda.empty_cache()
                if args.ensemble and len(members)>1:
                    for i,(sample,prob) in enumerate(zip(stream,np.mean(members,axis=0))):
                        scores.append(dict(**key,arm=arm+"_ensemble",seed=-1,step=i,
                                           **metrics(sample.s,sample.y,prob.argmax(0))))
                save()
            if not args.skip_warc:
                for arm in args.warc_arms.split(","):
                    if arm not in ("frozen","ttt","scratch"): raise ValueError(arm)
                    for seed in range(args.seeds if arm!="frozen" else 1):
                        for row in warc_scores(game,stream,arm,seed,args.device):
                            scores.append(dict(**key,**row))
                        save()
    save()
    df=pd.DataFrame(scores)
    group=["task","arm"] if "task" in df else ["game","arm"]
    metrics_cols=["changed_acc","false_edit","exact","change_f1"]
    if "session" in df:
        # Equal weighting per recording, then per game. Avoid long-recording bias.
        per=df.groupby(["game","session","arm","seed"])[metrics_cols].mean()
        paired=per.groupby(["game","session","arm"]).mean()
        print(paired.groupby(["game","arm"]).mean().round(4))
        pivot=paired["changed_acc"].unstack("arm")
        if "grow" in pivot:
            delta=pivot.apply(lambda col:pivot["grow"]-col)
            delta.to_csv(out/"paired_changed_accuracy.csv")
            print("Mean paired differences (grow - comparator):",delta.mean().round(4).to_dict())
    else:
        print(df.groupby(group)[metrics_cols].mean().round(4))
    print("\nPublished MechEval (NOT the replay proxy above):")
    print(pd.DataFrame(REFERENCE,index=["All_pct","EtC","not_reached_pct"]).T)
    if args.profile: profile_update(args.device,out,profile_samples)


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--suite",choices=("smoke","synthetic","arc"),default="synthetic")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--out",default="recursive_results")
    p.add_argument("--arms",default=",".join(ARMS))
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--seeds",type=int,default=3)
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--games",default="ls20,ka59,tr87")
    p.add_argument("--sessions",type=int,default=2)
    p.add_argument("--limit",type=int,default=48)
    p.add_argument("--data"); p.add_argument("--warc-root",default="/content/WARC")
    p.add_argument("--skip-warc",action="store_true")
    p.add_argument("--warc-arms",default="frozen,ttt,scratch")
    p.add_argument("--ensemble",action="store_true")
    p.add_argument("--hard-arc",action="store_true")
    p.add_argument("--profile",action="store_true")
    args=p.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("Select a GPU runtime")
    if args.device=="cuda": torch.set_float32_matmul_precision("high")
    else: torch.set_num_threads(min(4,torch.get_num_threads()))
    selftest(args.device); run(args)
