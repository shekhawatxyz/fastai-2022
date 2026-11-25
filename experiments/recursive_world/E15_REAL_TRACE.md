# E15 — Real-world diagnostic: no gain
Frozen protocol: [6e6f0b](https://github.com/shekhawatxyz/fastai-2022/blob/6e6f0b31ab5a23b7d7a1b254c2767a2fe603cef0/experiments/recursive_world/E15_PROTOCOL.md).
Source: [Shoshin traced ft09 gated seed 0](https://github.com/Sebastianrodaaa/shoshin/blob/main/results/demo/ft09_gated_s0.json), blob 06103407d1dfd5938ab1fd9c828d43477275b67a. 40 transitions, 35 with changes. Fold3 held-out pretrained WARC, CPU, minibatch 1, surprise-gated Muon/AdamW. Not full WARC TTT.
Executed an independent JavaScript port of E14 residual equations on the cached frames, preserving causal order, priors, role grouping, count weighting, expert losses and freeze at step 5. Teacher changed accuracy matched every defined upstream row to 1e-9. Port was not checked for bitwise parity with Python; ties can differ. No weights downloaded, no GPU run.

| Arm | Changed accuracy % | False edits % | Exact % | F1 |
|---|---:|---:|---:|---:|
| teacher | 68.5714 | .00427246 | 60 | .608333 |
| raw | 32.8571 | .00061035 | 10 | .383333 |
| coarse | 51.4286 | .00061035 | 42.5 | .491667 |
| fixed | 68.5714 | .00427246 | 60 | .608333 |
| dynamic | 68.5714 | .00366211 | 60 | .608333 |
| frozen | 68.5714 | .00366211 | 60 | .608333 |

Forecast dynamic-teacher +1 pp; measured zero. Gate failed (requires +3 pp). Dynamic and frozen identical on aggregate. Cannot compare directly with Shoshin's 61.1% full WARC six-game aggregate. E14's representation is local and action-ID-only; no click coordinates, object relations or event-level role bindings. Do not tune against this single trace and call it a held-out result.

Reproduction with existing Python (not independently executed here): download the source JSON to ft09.json beside e14_residual.py; run the following. Compare to JS measurements; report differences.
```python
import json
import numpy as np
from e14_residual import Residual, metrics
d=json.load(open('ft09.json'))
m=Residual()
rows={}
def decode(x):
    return np.array([int(c,16) for c in x],dtype=np.uint8).reshape(64,64)
for row in d['rows']:
    s,p,y=map(decode,(row['prev'],row['pred'],row['real']))
    pred,cache=m.predict(s,p,row['action'][0])
    for k,v in pred.items():rows.setdefault(k,[]).append(metrics(s,y,v))
    m.observe(cache,y)
for k,v in rows.items():print(k,np.nanmean(v,axis=0))
```

Next design: action-relative roles, not static color equivalence. Preserve strong neural predictor, full TTT and this adapter as baselines. See architecture contract for future experiment. Source for baseline aggregate: https://github.com/Sebastianrodaaa/shoshin/blob/main/docs/REPORT.md .