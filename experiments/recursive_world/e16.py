"""E16: exact event-word induction; see frozen E16_PROTOCOL.md."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import itertools as it
import numpy as np
from time import perf_counter

W=[w for k in (1,2,3) for w in it.product(range(2),repeat=k)]
V=[w for k in (1,2) for w in it.product(range(3),repeat=k)]
T=[(W.index((0,1)),V.index((2,0))),(W.index((1,0)),V.index((1,2)))]

class Space:
    def __init__(self,n):
        self.n=n; self.p=np.array(list(it.permutations(range(n)))); self.g=len(self.p)
        lookup={tuple(p):i for i,p in enumerate(self.p)}
        self.mul=np.array([[lookup[tuple(b[a])] for b in self.p] for a in self.p])
        a,b=np.indices((self.g,self.g)); self.a=a;self.b=b
        def word(w,ops):
            p=ops[w[0]]
            for j in w[1:]:p=self.mul[p,ops[j]]
            return p
        self.c=np.stack([word(w,[a,b]) for w in W],-1)
        self.d=np.stack([np.broadcast_to(word(w,[a[:,:,None],b[:,:,None],self.c]),self.c.shape) for w in V],-1)
        self.ix=np.broadcast_arrays(a[:,:,None,None],b[:,:,None,None],self.c[:,:,:,None],self.d)
        x=np.array(list(it.product(range(2),repeat=n))); self.x=x
        self.y=(x[:,self.p]*(1<<np.arange(n))).sum(-1).T
    def sample(self,rng,family):
        a,b=rng.integers(self.g,size=2)
        if family=='independent': return a,b,*rng.integers(self.g,size=2)
        j=rng.integers(2);w,v=T[j];v=T[1-j][1] if family=='shifted' else v
        return a,b,self.c[a,b,w],self.d[a,b,w,v]
    def pretrain(self,rng):
        counts=np.ones((len(W),len(V)))
        for _ in range(96):
            a,b,c,d=self.sample(rng,'matched');ok=(self.c[a,b,:,None]==c)&(self.d[a,b]==d)
            counts+=ok/ok.sum()
        return counts/counts.sum()
    def obs(self,rng,p,k):
        q=rng.integers(len(self.x),size=k); return q,self.y[p,q]
    def mask(self,obs):
        q,y=obs;return (self.y[:,q]==y).all(1)
    def scores(self,probs,p,obs):
        unseen=np.setdiff1d(np.arange(len(self.x)),obs[0])
        if not len(unseen):return np.nan,np.nan
        pred=np.array([np.bincount(self.y[:,q],weights=probs,minlength=len(self.x)).argmax() for q in unseen]);return np.mean(pred==self.y[p,unseen]),float(probs.argmax()==p)
    def posterior(self,masks,prior):
        m=masks[0][:,None,None,None]*masks[1][None,:,None,None]*masks[2][self.c][:,:,:,None]*masks[3][self.d]
        w=m*prior[None,None,:,:]/self.g**2; z=w.sum()
        if z==0:return z,None,None
        wc=np.bincount(self.ix[2].ravel(),weights=w.ravel(),minlength=self.g)/z
        wd=np.bincount(self.ix[3].ravel(),weights=w.ravel(),minlength=self.g)/z
        return z,wc,wd
    def trial(self,rng,family,prior):
        truth=self.sample(rng,family);obs=[self.obs(rng,p,1 if i<3 else 4) for i,p in enumerate(truth)]
        fac=prior.sum(0)[None,:]*prior.sum(1)[:,None];uni=np.full_like(prior,1/prior.size);rows=[];frozen=None
        for k in (0,1,2,4):
            o=obs[:3]+[(obs[3][0][:k],obs[3][1][:k])];m=[self.mask(v) for v in o];ind=[v/v.sum() for v in m];z0=np.prod([v.mean() for v in m]);preds=[]
            for pr in (uni,fac,prior):
                z,c,d=self.posterior(m,pr)
                if c is None:c,d=ind[2:]
                preds.append((c,d))
            c,d=preds[-1];alpha=z/(z+z0);rob=(alpha*c+(1-alpha)*ind[2],alpha*d+(1-alpha)*ind[3])
            if frozen is None:frozen=rob[0].copy()
            arms=[tuple(ind[2:]),*preds,rob,(frozen,rob[1])]
            rows.append([[*self.scores(d,truth[3],o[3]),*self.scores(c,truth[2],o[2])] for c,d in arms])
        return rows

def main():
    start=perf_counter()
    for n,seed in [(3,16161),(4,16262)]:
        source=Space(n);target=Space(7-n);rng=np.random.default_rng(seed);prior=source.pretrain(rng)
        print('direction',n,'->',7-n,'hypotheses',target.d.size,'top prior',[(W[i],V[j],round(prior[i,j],3)) for i,j in zip(*np.unravel_index(np.argsort(prior.ravel())[-3:][::-1],prior.shape))],flush=True)
        for family in ('matched','shifted','independent'):
            a=np.array([target.trial(rng,family,prior) for _ in range(80)]);print(family,'D accuracy k=0,1,2,4 arms ind/uniform/factor/joint/robust/frozen',np.round(np.nanmean(a[:,:,:,0],0)*100,2),flush=True)
            for name,d in [('joint-uniform D0',a[:,0,3,0]-a[:,0,1,0]),('joint-factor D0',a[:,0,3,0]-a[:,0,2,0]),('robust-frozen C2',a[:,2,4,2]-a[:,2,5,2]),('robust-independent D2',a[:,2,4,0]-a[:,2,0,0])]:
                d=d[np.isfinite(d)];boot=np.random.default_rng(44).choice(d,(2000,len(d)),replace=True).mean(1);print(name,round(d.mean()*100,3),np.round(np.quantile(boot,[.025,.975])*100,3),flush=True)
    print('wall',perf_counter()-start)
if __name__=='__main__':main()
