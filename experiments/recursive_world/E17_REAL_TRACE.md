# E17 — whole-event primitive: real trace development diagnostic

[Protocol frozen before execution](https://github.com/shekhawatxyz/fastai-2022/blob/9ac452f8d88cbe2c2906f2ced94b07dd4633249b/experiments/recursive_world/E17_PROTOCOL.md). [Exact executed source](https://github.com/shekhawatxyz/fastai-2022/blob/d3f663a2c051b0b3bc48516d9e986ecab21f9135/experiments/recursive_world/e17.mjs). No GPU.

```
observed transition -> whole edit + source context
                     -> candidate literal/role-binding schemas
old events ----------> re-index evidence by candidate schema
current scene + action -> uniquely bound whole-event proposals
WARC prediction + proposals -> prequential selection -> next frame
reveal outcome -> score then revise evidence
```

E17 is a bounded event-template retrieval implementation, not the complete proposed graph, temporal event discovery or higher-order induction. It uses two hypotheses about the same old events: literal source context and bijective color-role bindings. It does not read game source or infer from future observations. Fixed 16x16 maximum patch and 1-cell halo. Require two agreeing prior witnesses and a strict plurality; ambiguous placements abstain. No parameter tuning after scores.

## Measured result
Source: [Shoshin ft09 gated CPU trace](https://github.com/Sebastianrodaaa/shoshin/blob/main/results/demo/ft09_gated_s0.json), blob 06103407d1dfd5938ab1fd9c828d43477275b67a. Already inspected in E15, so development only. 40 transitions, 35 changed. Teacher accuracy matches upstream on every defined step.

| Arm | Changed accuracy | False edits | Exact | F1 |
|---|---:|---:|---:|---:|
| WARC gated (cached) | 68.5714% | .00427246% | 60% | .608333 |
| Literal templates + teacher fallback | 68.5714% | .00427246% | 60% | .608333 |
| Role-bound templates + teacher fallback | 68.5714% | .00427246% | 60% | .608333 |
| Prequential selector | 68.5714% | .00427246% | 60% | .608333 |

36 templates accepted; literal proposal coverage 18/40, role coverage 0/40. Both net exact wins zero. Forecast +2pp, actual 0: development gate FAIL. Role coverage is zero after requiring unique placement and two agreeing witnesses; the aggregate counters do not isolate which condition is responsible. Millisecond wall-clock instrumentation in JS execution: prediction 495, template construction 9, scoring 71 across 40 frames. Not a kernel profiler nor GPU performance forecast.

Reproduce on a machine with network and Node:
```sh
curl -L https://raw.githubusercontent.com/Sebastianrodaaa/shoshin/main/results/demo/ft09_gated_s0.json -o ft09.json
node e17.mjs ft09.json
node --prof e17.mjs ft09.json
```

## What to preserve and what to change
E1–E17: the strongest toy result is E16's bidirectional inference; E14 and E15 show that raw or coarse roles cannot guarantee a useful intervention. E17 demonstrates ambiguity rather than a solution. The next primitive must describe event-relative participant relations, including temporal purpose and relational context, not a spatial patch or free color permutation. Retain qualitative uncertainty over event identity before deriving precise dynamics. Use dynamic scope, structured event proposals, a reusable library of event schemas, and neural fallback together. Compare against equally expressive ordinary schema induction; pretrain and ensemble the proposal policy only if data supports it. The decisive evidence is held-out cross-mechanic/changed-world transfer, not number of graph nodes or recursive calls.

## Literature and benchmark decision
- Bhartṛhari [VP 3.8.4](https://www.wisdomlib.org/hinduism/book/vakyapadiya-of-bhartrihari/d/doc1337253.html), [3.8.5 with commentary](https://www.wisdomlib.org/hinduism/book/vakyapadiya-of-bhartrihari/d/doc1337254.html), [3.7.92 with commentary](https://www.wisdomlib.org/hinduism/book/vakyapadiya-of-bhartrihari/d/doc1337177.html): whole action across moments and refinement of instruments toward ends. These are philosophical/grammatical inspirations, not modern algorithms.
- [Human ARC think-aloud](https://arc3-human-think-aloud.github.io/): qualitative representations and uncertainty are observed more frequently in humans, but authors explicitly note that precise mechanistic models are effective in ARC. Qualitative abstraction must earn its predictive/planning utility.
- [Kepler](https://github.com/Cveinnt/kepler) reports 100 on public replay, explicitly not official first exposure; [Pathbreaker](https://github.com/gzatravkin/pathbreaker-arc-agi-3) rank 6 public competition at close uses pretrained Qwen, expert switching, optimized serving, agent writes code. These are different metrics from WARC. Read their README and reports; not all Kaggle notebook cells inspected. WARC and Shoshin implementation files inspected in earlier experiments.
- [AutumnBench](https://proceedings.mlr.press/v306/warrier26a.html) (43 worlds, 129 queries) separates acquisition from transfer evaluation; [AutoWorldModel-Bench](https://autoworldmodelbench.github.io/) supplies structured entity states with Dreamer, AR-Transformer, D3PM, MaskGIT starters. Useful independent checks of generality; do not conflate their metrics with ARC.
- [AgentOWL](https://www.cs.cornell.edu/~wp237/papers/piriyakulkij2026agentowl.pdf), [REFACTOR-VLA](https://arxiv.org/abs/2609.01215), [PoE-World](https://papers.neurips.cc/paper_files/paper/2025/file/262dd62fd1bbb30d6a6b4d578f5e65ff-Paper-Conference.pdf): temporal abstraction, reusable typed motor programs, compositional dynamics are existing work. A complete prior-art check remains necessary.

## Data and scale gate
The CPU environment cannot resolve GitHub or Hugging Face to fetch weights/data. Connector exposes only one frame-level traced Shoshin run; summary logs cannot be substituted. Need official human recording archive and WARC held-out checkpoints or complete causal cached predictions from full TTT on independent games. For genuine benchmark SOTA require official evaluator, comparable protocol, full baseline, frozen splits, compute and uncertainty. No SOTA achieved; no GPU scaling warranted. A small diagnostic producing real pretrained predictions is distinct from scaling our model.
