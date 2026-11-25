# E21 — decision and measured CPU results
2026-11-25. Preserve all previous research. The original local protocol was frozen before execution with a SHA256 record; this GitHub summary is retrospective, not preregistration.

E21 compared an ALPaCA-like learned Bayesian response field trained on ordinary endpoint residuals vs action-effect prediction, with same source episodes, initializations, optimizer and inference. Two simulated physical systems, two training seeds, AdamW .003 cosine; source paired interventions available to all arms; target alternative outcomes hidden. Normalized action-effect MSE is primary (lower better). Comparators include fixed features, GRU and additive residual reuse. This is a local simulation, not an official SOTA reproduction.

| Gain of effect-trained model (normalized MSE percentage points, positive is better) | System 0 | System 1 |
|---|---:|---:|
| vs residual reuse, in-distribution | 3.588 | 3.691 |
| vs ordinary learned field, in-distribution | 0.126 [0.074, 0.182] | 0.090 [0.029, 0.152] |
| vs fixed effect features, in-distribution | 0.646 | 0.654 |
| vs GRU effect model, in-distribution | 3.594 | 3.653 |
| vs ordinary field, fresh disturbances | 0.070 [-0.055, 0.205] | 0.016 [-0.121, 0.149] |
| vs ordinary field, held-out saturation | 0.123 [0.009, 0.236] | 0.143 [-0.003, 0.292] |

Frozen gate >=1 pp against ordinary field, fixed and GRU in both systems: FAIL. Small improvement over ordinary field, not evidence of SOTA. CPU wall 20.15s; batch64 timing 4.16ms covers all arms, not candidate-only. Local notebook validated against nbformat and Python AST; complete notebook execution not independently repeated. Local research bundle contains code, protocol, logs and results; this commit records findings, not those full files. No code run on GitHub.

E18–21 pattern: flexible latent slots fail to learn the operational structure; forcing shared context hurts fresh disturbances; handcrafted features remain strong; training on action effects improves a matched learned field only modestly. Do not attribute those outcomes to cultural theory or retrofit a new explanation as a success.

Research constraint going forward: read the actual strongest implementation and reproduce its official metrics first. Keep its representation, pretrained weights, optimizer, inference and evaluation. Select at most one or two lightweight heuristic changes based on a demonstrated failure mode and a computational rationale; no obligation to map ancient concepts one-to-one. If an ordinary ML explanation yields the same intervention, use it. Freeze one hypothesis and comparator, report failed results, stop toy churn. Real benchmark evidence, not a toy gain, determines whether further GPU scaling is warranted.
