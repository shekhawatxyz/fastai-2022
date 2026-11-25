# Self-reorganizing relational world model — architecture contract

2026-11-25. Proposed, not validated. Preserve E1–E9 and all negative results.

## Hypothesis
Learning changes not merely a prediction but the language in which future predictions and learning are expressed. A newly discovered relationship becomes a reusable entity; existing observations and concepts can be reinterpreted through it. The test is cross-mechanic transfer before seeing the target mechanic, not repeated execution of a fixed network.

```
raw experience -> pretrained encoder -> objects/events/relations -> dynamics -> prediction
                         ^                     |                      |
                         |           propose abstraction             |
                         |                     v                      |
                         +--- reinterpret <--- concept graph <--------+
                                      ^               |
                                      +-- revise <--- contradiction / predictive validation
```

1. Keep immutable observations and actions. Use pretrained WARC/ViT for perception and predictive fallback; do not force all information through symbolic slots.
2. Represent a concept as a parameterized relation with executable or neural semantics, bindings, dependencies, evidence and uncertainty. Relations and compositions may themselves become nodes. Identity is not a permanent semantic label.
3. Discover candidate abstractions by compression of recurring relational prediction errors and successful dynamics; use anti-unification/typed program synthesis for the first toy, learned proposal network later if it improves search. Candidate discovery must go beyond selecting an enumerated dictionary. Keep multiple plausible interpretations.
4. Re-encode old experience when a relation is added/revised, propagate changes only through dependent concepts, and test predictive effects. Inherited scope and exceptions are inferred, not permanent hand-assigned categories. Admit a change only if causal predictive improvement or description-length reduction pays for complexity; preserve uncertainty and a neural fallback. Historical prequential scores are immutable.
5. Predict next-frame and future abstract relational state with action-conditioned dynamics. Keep ordinary gradient updates; train a shared proposal/revision policy across source environments only after the CPU mechanism wins. Use action-conditioned latent prediction with independently anchored targets to prevent representation collapse. Planning is downstream, not the first experiment.

## Why this is not blind recursion
Schmidhuber's History Compressor recursively abstracts unpredictable events; H-JEPA predicts at multiple timescales. DreamCoder invents reusable program abstractions; OPINE-World revises object ontologies. Our proposed distinguishing test is learned relational abstractions which become new operands of induction and retroactively change evidence indexing, yielding measurable transfer to different mechanics with zero/few own exposures. None of these individual ingredients establishes novelty.

Pāṇini/Kātyāyana/Patañjali's mutual dependencies, rule conditions and exceptions, and Bhartṛhari's whole-to-part abstraction motivate the hypothesis. These texts do not prescribe an ML algorithm. See Bronkhorst (2004) https://www.academia.edu/3279193/ and https://plato.stanford.edu/archives/spr2021/entries/language-india/ .

## Test before scale
Generate two independent small action-conditioned worlds with held-out relational compositions and confounding appearance. Hide semantic labels. Same transitions, source data, candidate budget and evaluation protocol for all arms: raw replay, exact full-history latent inference/strong program synthesis, fixed ontology, dynamic ontology without retroactive reinterpretation, dynamic ontology with reinterpretation, pretrained neural dynamics and neural+hybrid. Measure target mechanic accuracy at zero and 1–4 exposures, held-out composition and appearance generalization, recovery from exceptions, false edits, multi-step predictive error, total search/training cost. Include a no-shared-structure control. Freeze seeds and gates in a separate commit before running. Profiling verifies correctness and avoids bottlenecks; it is not the objective.

## Real target
WARC/MechEval reports 69.1% all exposures, 41.9% exposures 1–4, 74 median exposures to criterion, 0.11% false edits. Rank-64 LoRA improves 1–4 by 3.8 points; fully updated GRU reports 70.0% on one seed. WARC paper notes that withholding a mechanic's transitions leaves accuracy near frozen; new object clicks are especially difficult. Target cross-mechanic transfer, not the saturated public ARC game-completion score. Ambitious success thresholds (not forecasts): >=80% all exposures, >=60% early, <=30 median exposures, <=0.11% false edits, comparable pretrained parameters and compute. Require official scorer/splits and matched baselines; absent scorer, label replay proxy only. Run a small real-data diagnostic against full WARC TTT before expensive scaling, after CPU gates pass.

References: https://arxiv.org/abs/2611.12324 ; https://arxiv.org/abs/2607.01531 ; https://arxiv.org/abs/2006.08381 ; https://papers.nips.cc/paper/1991/hash/2bb232c0b13c774965ef8558f0fbd615-Abstract.html ; https://openreview.net/pdf/315d43ba26f55357a84cec9a7ed15a6610094f79.pdf .

Status: architecture not fully implemented; E10–E12 work was interrupted before publication, so no verified profiling or GPU claim from this specification.
