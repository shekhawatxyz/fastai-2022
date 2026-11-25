"""Causal residual: no target at predict; roles re-index immutable counts."""
import numpy as np
from time import perf_counter

class Residual:
    def __init__(self, colors=16, actions=8, noise=.05):
        self.c,self.a,self.noise=colors,actions,noise
        self.o=6+colors
        self.count=np.zeros((colors,actions,16,self.o),np.float64)
        self.n=np.zeros((colors,actions),np.int64)
        self.role=np.arange(colors); self.frozen=self.role.copy(); self.t=0
        self.loss=np.zeros((actions,5));self.loss_f=np.zeros((actions,5))
    def features(self,s,p,a):
        s=np.asarray(s,dtype=int);p=np.asarray(p,dtype=int);z=np.pad(s,1,mode='edge')
        v=np.stack([s,z[1:-1,:-2],z[1:-1,2:],z[:-2,1:-1],z[2:,1:-1]],-1).reshape(-1,5)
        c=v[:,0];r=((v[:,1:]==c[:,None])*(1<<np.arange(4))).sum(1)
        e=np.concatenate([p.reshape(-1,1),v,np.broadcast_to(np.arange(self.c),(len(c),self.c))],1)
        return c,r,e
    def _p(self,v,e):
        # Uniform effect prior; exact noise-model posterior within each scope.
        w=v*np.log((1-self.noise)/(self.noise/(self.c-1)))
        w=np.exp(w-w.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        out=np.zeros((len(e),self.c));np.add.at(out,(np.arange(len(e))[:,None],e),w)
        return out*(1-self.noise)+(1-out)*self.noise/(self.c-1)
    def predict(self,s,p,a):
        c,r,e=self.features(s,p,a);k=int(a[0] if np.ndim(a) else a)
        raw=self.count[:,k];glob=raw.sum((0,1));context=raw.sum(0);coarse=raw.sum(1)
        def pooled(role):
            z=np.zeros_like(raw);np.add.at(z,role,raw);return z[role[c],r]
        teacher=np.eye(self.c)[e[:,0]]*(1-self.noise)+(1-np.eye(self.c)[e[:,0]])*self.noise/(self.c-1)
        ps=[teacher,self._p(raw[c,r],e),self._p(np.broadcast_to(glob,(len(c),self.o)),e),self._p(context[r],e),self._p(pooled(self.role),e)]
        pf=ps[:4]+[self._p(pooled(self.frozen),e)]
        def mix(bank,loss):
            w=np.exp(-loss+loss.min());w/=w.sum();return np.einsum('i,inc->nc',w,np.array(bank))
        probs=dict(teacher=teacher,raw=ps[1],coarse=self._p(coarse[c],e),fixed=mix(ps[:4],self.loss[k,:4]),dynamic=mix(ps,self.loss[k]),frozen=mix(pf,self.loss_f[k]))
        return {name:q.argmax(1).reshape(s.shape) for name,q in probs.items()},(c,r,e,k,ps,pf)
    def observe(self,cache,y):
        c,r,e,k,ps,pf=cache;y=np.asarray(y).ravel();idx=np.arange(len(y))
        # Per-frame log loss, not independent evidence for each neighboring cell.
        self.loss[k]-=np.log(np.stack(ps)[:,idx,y]).mean(1)
        self.loss_f[k]-=np.log(np.stack(pf)[:,idx,y]).mean(1)
        # Normalize sufficient statistics per color/context per frame.
        key=c*16+r;tot=np.bincount(key,minlength=self.c*16)
        weight=1/np.maximum(tot[key],1)
        np.add.at(self.count[:,k],(c,r),(e==y[:,None])*weight[:,None])
        np.add.at(self.n[:,k],c,1)
        stats=self.count.sum(2);sig={};role=np.arange(self.c)
        for i in range(self.c):
            known=(self.n[i]>=8);v=stats[i];best=v.argmax(1)
            top=np.sort(v,axis=1)[:,-2:];certain=known&(top[:,1]>top[:,0]+.5)
            if not certain.any():continue
            signature=tuple(np.where(certain,best,-1))
            role[i]=sig.setdefault(signature,i)
        self.role=role;self.t+=1
        if self.t<=5:self.frozen=role.copy()

def stream(seed,kind,n=20):
    rng=np.random.default_rng(seed);colors=8;role=rng.permutation(colors)%2;random=rng.integers(0,5,(colors,2,16))
    m=Residual(colors,2)
    for t in range(n):
        s=rng.integers(0,colors,(8,8));a=t%2
        z=np.pad(s,1,mode='edge');p=z[1:-1,:-2] if a==0 else z[1:-1,2:]
        c,r,e=m.features(s,p,a)
        if kind=='shared': op=np.full(len(c),a+2)
        elif kind=='roles':op=np.where(role[c]==0,a+2,a+4)
        else:op=random[c,a,r]+1
        y=e[np.arange(len(c)),op];noisy=rng.random(len(y))<.05;y=np.where(noisy,(y+rng.integers(1,colors,len(y)))%colors,y)
        yield s,p,a,y.reshape(s.shape)

def metrics(s,y,p):
    ch=s!=y;edit=s!=p
    return [(p==y)[ch].mean() if ch.any() else np.nan,edit[~ch].mean() if (~ch).any() else np.nan,float(np.array_equal(y,p)),2*(edit&ch).sum()/max(1,(edit.sum()+ch.sum()))]

def run():
    import pandas as pd
    rows=[];t0=perf_counter()
    for kind in ('shared','roles','random'):
        for seed in range(14100,14130):
            m=Residual(8,2)
            for t,(s,p,a,y) in enumerate(stream(seed,kind),1):
                start=perf_counter();pred,cache=m.predict(s,p,a);pt=perf_counter()-start
                if t==1:assert all(np.array_equal(v,m.predict(s,p,a)[0][k]) for k,v in pred.items())
                start=perf_counter();m.observe(cache,y);ut=perf_counter()-start
                for name,v in pred.items():rows.append((kind,seed,t,name,*metrics(s,y,v),pt,ut,len(set(m.role))))
    d=pd.DataFrame(rows,columns='kind seed step arm changed false exact f1 predict_s update_s roles'.split())
    e=d.query('5<=step<=12');print(e.groupby(['kind','arm'])[['changed','false','exact','f1']].mean().round(4))
    for kind in ('shared','roles','random'):
        p=e[e.kind==kind].groupby(['seed','arm']).changed.mean().unstack();delta=p.dynamic-p.fixed
        boot=np.random.default_rng(42).choice(delta,(3000,len(delta)),replace=True).mean(1)
        print(kind,'dynamic-fixed',round(delta.mean(),5),np.quantile(boot,[.025,.975]),'dynamic-frozen',round((p.dynamic-p.frozen).mean(),5))
    print('wall seconds',perf_counter()-t0,'median predict/update',d[['predict_s','update_s']].median().to_dict())
    d.to_csv('e14_results.csv',index=False)

if __name__=='__main__':
    import cProfile,pstats
    pr=cProfile.Profile();pr.enable();run();pr.disable();pstats.Stats(pr).strip_dirs().sort_stats('cumulative').print_stats(12)
