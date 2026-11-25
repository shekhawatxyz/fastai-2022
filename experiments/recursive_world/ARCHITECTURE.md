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


## Amendment after E15: event-relative conceptual semantics (proposed, not measured)

E15: the E14 semantic residual tied cached pretrained surprise-gated WARC at 68.57% changed accuracy on the released 40-step ft09 trace; no gain from continuing revision. Its concept is predominantly a static color-role signature. Do not extrapolate to full TTT or MechEval. See E15_REAL_TRACE.md. Preserve the residual as an ablation.

Textual grounding: Pāṇini 1.4.23–55 assigns kārakas in relation to action, rather than identifying them with surface case markers (https://web.stanford.edu/class/linguist289/encyclopaedia001.pdf). Bhartṛhari Vākyapadīya 3.7.63 and 3.7.104 explicitly address changes of role with context (https://www.wisdomlib.org/hinduism/book/vakyapadiya-of-bhartrihari/d/doc1337149.html and https://www.wisdomlib.org/hinduism/book/vakyapadiya-of-bhartrihari/d/doc1337188.html). 1.83–85 and 1.124 motivate provisional distinctions and mutual determination. These are interpretations of texts, not claims of modern algorithms in antiquity.

### Concrete computational interface
- Raw event: (frame history, action including click position, outcome). An event candidate is an uncertain set of entities, state changes, and bindings of participant variables to entity IDs. Do not pass true future outcome into inference at prediction.
- Concept: a parameterized predicate or transition schema with typed variables, a guard, effect, dependencies, evidence and exceptions. A participant's role is a binding in an event schema, not its immutable type. A schema can take other schemas as arguments. No hard-coded semantic names such as lock, agent, or authorization.
- Neural branch: frozen or test-time-trained WARC for prediction and embeddings. Source-trained proposal network may suggest entities, bindings and schema edits; it is not given test labels. Neural fallback remains valid even when symbolic hypotheses fail.
- Initial search: infer candidate transitions from raw frame differences on observed events only. Anti-unify verified event descriptions (replace differing substructures by shared variables), use recurrent effects to propose new predicates and schemas, including abstractions over existing schemas. Preserve both independently explained and shared explanations. Do not assume causality from co-occurrence.
- Two-way semantics: if a proposed schema predicts more than one previously independent event family, refactor their descriptions to bind the new variables, recheck old raw observations, and reconsider affected candidate programs. Later observations can support, split, or delete a schema. Keep a beam of rival theories and immutable original observations; never rewrite earlier prediction scores. Cache unaffected dependencies.
- Prediction: infer event bindings from observed history and current action, then marginalize viable schema effects alongside the neural distribution; abstain when contradictory or unidentifiable. Learn the proposal policy on independent source tasks only, measure pretraining cost. Learning can alternate between prediction, inference, and structural search, not a fixed number of recurrent passes.
- Admission: rank candidates by description length (library + event descriptions + residual coding cost), compare predictive accuracy on subsequent causal observations and held-out development tasks. Charge all proposals, refactoring, pretraining and ensemble compute. A shorter explanation without transfer is not a success.

### Strongest comparison
Schema Networks already support causal zero-shot transfer (https://proceedings.mlr.press/v70/kansky17a.html); dynamic NRI learns time-varying relations (https://arxiv.org/abs/1802.04687 and https://openaccess.thecvf.com/content_CVPR_2020/html/Graber_Dynamic_Neural_Relational_Inference_CVPR_2020_paper.html); DreamCoder learns a library and refactors programs (https://pubmed.ncbi.nlm.nih.gov/37271169/); OPINE-World revises programmatic ontologies (https://arxiv.org/abs/2607.01531). None of event roles, graph inference, hierarchical compression or refactoring alone is novel. The distinguishing experiment must test online schema-of-schema invention plus causal retrospective re-binding against an equally expressive library learner with conventional belief revision, not just raw replay or a fixed neural model.

### Next minimal test, before training
Two independent event generators with participant role reversals, visually randomized objects and held-out combinations of known mechanics. The first can use spatial movement/containment; the second non-spatial resource transfer/authorization. Hide role and schema identities. Train/pretrain on source worlds; freeze test worlds, seeds, budgets and gates before running. Score exact next state, held-out composition and appearance, zero/few own-exposure transfer, and effect of novel evidence on untouched earlier events. Include unrelated and noisy cases. Compare flat synthesis, schema learner with fixed event language, dynamic roles, growing library without retrospective re-binding, and full model; add strong joint inference/refactoring baseline, plus neural and hybrid arms with identical source data. Only scale if the full mechanism beats the strongest appropriate CPU comparator, not merely if a toy matches our intuition. Published official WARC numbers remain reference only until scorer, splits and matched runs are available.
