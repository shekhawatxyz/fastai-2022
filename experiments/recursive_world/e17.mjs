function e17(trace) {
  const W=64,N=W*W, bank=[];
  const decode=s=>Uint8Array.from(s,x=>parseInt(x,16));
  const same=(a,b)=>a.every((v,i)=>v===b[i]);
  function template(s,y,a) {
    let d=[];for(let i=0;i<N;i++)if(s[i]!==y[i])d.push(i);
    const click=a[0]===6&&a[1]>=0&&a[2]>=0;
    if(!d.length&&!click)return null;
    let points=d.map(i=>[i%W,Math.floor(i/W)]);if(click)points.push([a[1],a[2]]);
    let l=Math.max(0,Math.min(...points.map(p=>p[0]))-1),r=Math.min(63,Math.max(...points.map(p=>p[0]))+1),t=Math.max(0,Math.min(...points.map(p=>p[1]))-1),b=Math.min(63,Math.max(...points.map(p=>p[1]))+1);
    if(r-l>=16||b-t>=16)return null;
    const patch=[];for(let j=t;j<=b;j++)for(let i=l;i<=r;i++)patch.push([i-l,j-t,s[j*W+i]]);
    return {id:a[0],click,ax:click?a[1]-l:0,ay:click?a[2]-t:0,w:r-l+1,h:b-t+1,patch,edits:d.map(i=>[i%W-l,Math.floor(i/W)-t,y[i]])};
  }
  function match(s,a,p,role) {
    if(a[0]!==p.id)return null;
    let found=null;
    const x0=p.click?a[1]-p.ax:0,y0=p.click?a[2]-p.ay:0;
    const x1=p.click?x0:W-p.w,y1=p.click?y0:W-p.h;
    if(x0<0||y0<0||x1>W-p.w||y1>W-p.h)return null;
    for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
      const map=new Int8Array(16).fill(-1),inv=new Int8Array(16).fill(-1);
      let ok=true;
      for(const [dx,dy,c] of p.patch){
        const q=s[(y+dy)*W+x+dx];
        if(!role){if(q!==c){ok=false;break}}
        else if((map[c]>=0&&map[c]!==q)||(inv[q]>=0&&inv[q]!==c)){ok=false;break}
        else{map[c]=q;inv[q]=c}
      }
      if(!ok)continue;
      const edits=[];
      for(const [dx,dy,c] of p.edits){
        if(role&&map[c]<0&&inv[c]>=0){ok=false;break}
        const i=(y+dy)*W+x+dx,q=role&&map[c]>=0?map[c]:c;
        if(s[i]!==q)edits.push([i,q]);
      }
      if(!ok)continue;
      const key=JSON.stringify(edits);
      if(found&&found.key!==key)return null;
      found={key,edits};
    }
    return found;
  }
  function proposal(s,a,role){
    let votes=new Map();
    for(const p of bank){
      let m=match(s,a,p,role);if(!m)continue;
      let v=votes.get(m.key);if(v)v.n++;else votes.set(m.key,{...m,n:1});
    }
    const v=[...votes.values()].sort((a,b)=>b.n-a.n);
    if(!v.length||v[0].n<2||(v.length>1&&v[0].n===v[1].n))return null;
    let out=s.slice();for(const [i,c] of v[0].edits)out[i]=c;return out;
  }
  function score(s,y,p){
    let ch=0,ok=0,un=0,fe=0,ed=0,both=0,exact=1;
    for(let i=0;i<N;i++){const c=s[i]!==y[i],e=s[i]!==p[i];if(c){ch++;ok+=p[i]===y[i]}else{un++;fe+=e}ed+=e;both+=e&&c;if(p[i]!==y[i])exact=0;}
    return [ch?ok/ch:null,un?fe/un:null,exact,2*both/Math.max(1,ed+ch)];
  }
  let stats=[0,0],coverage=[0,0],rows=[],time={template:0,predict:0,score:0};
  for(const row of trace.rows){
    const s=decode(row.prev),p=decode(row.pred),a=row.action;
    let start=Date.now(),l=proposal(s,a,false),r=proposal(s,a,true);time.predict+=Date.now()-start;
    const possible=[l,r].map((q,i)=>({q,i})).filter(v=>v.q&&stats[v.i]>0).sort((a,b)=>stats[b.i]-stats[a.i]);
    const chosen=possible[0]?.q??p;
    // Target decoded only after predictions committed.
    const y=decode(row.real);start=Date.now();
    const sc=[p,l??p,r??p,chosen].map(q=>score(s,y,q));
    if(row.acc!==null&&row.acc!==undefined&&Math.abs(sc[0][0]-row.acc)>1e-9)throw Error("teacher score mismatch");
    rows.push(sc);
    for(const [i,q] of [l,r].entries())if(q){coverage[i]++;stats[i]+=Number(same(q,y))-Number(same(p,y));}
    time.score+=Date.now()-start;
    start=Date.now();const ev=template(s,y,a);if(ev)bank.push(ev);time.template+=Date.now()-start;
  }
  const arms=["teacher","literal","role","selector"];
  return {transitions:rows.length,changed:rows.filter(r=>r[0][0]!==null).length,templates:bank.length,coverage,netExactWins:stats,profile_ms:time,results:arms.map((arm,i)=>({arm,mean:[0,1,2,3].map(j=>{let a=rows.map(r=>r[i][j]).filter(v=>v!==null);return a.reduce((a,b)=>a+b,0)/a.length})}))};
}
export {e17};
if (process.argv[1]?.endsWith('e17.mjs')) { const fs=await import('node:fs/promises'); const d=JSON.parse(await fs.readFile(process.argv[2],'utf8'));console.log(JSON.stringify(e17(d),null,2)); }
