# E24 — decision-directed probing
2026-11-25. Retrospective log; local pre-run protocol hash verified. No code run on GitHub. Preserve E1–E23.
Sources: https://people.idsia.ch/~juergen/FKI-149-91ocr.pdf ; https://arxiv.org/abs/0709.0674 ; https://arxiv.org/html/2611.24778v1 (Appendix C, Table 16); upstream planning.py inspected. Learning progress is not synonymous with useful control; knowledge gradient is an established acquisition method.
Frozen test: 5120 independent numerical boat events, five regimes, shared E23 trained two-MLP mean and source-selected GP (constant .9, periodic RBF length 1), 16 candidate actions, same initial probe, two adaptive probes, final recommendation. Same event reset. Goal sampled independently. Primary mean squared decision regret relative to best candidate, secondary threshold success and prediction MSE.
| Arm | Regret | Success | MSE |
|---|---:|---:|---:|
| Last residual | .001290939 | .399023 | .013064013 |
| Random GP | .005840269 | .378711 | .005004247 |
| Greedy GP | .000345482 | .399219 | .006722927 |
| Max variance | .005103122 | .380469 | .003664678 |
| Integrated variance | .004972502 | .382422 | .003684587 |
| Top-two contrast | .001832689 | .391992 | .005631276 |
| Knowledge gradient | .001043333 | .398242 | .006215143 |
Gate FAIL: KG regret 202% higher than best GP comparator greedy; paired best-minus-KG CI [-.0010173,-.0004166] (descriptive due to control selection). Greedy 73.24% less regret than last but success effectively unchanged. KG acquisition/evaluation timed 4.823s vs greedy .132s in same run (excludes shared mean training and simulation); total wall 24.85s. Initial shape error repaired before scored run; retained locally. Notebook syntax/nbformat validated, companion script executed; notebook itself not independently rerun.
Limit: diagnostic has three observed actions plus final recommendation, not the official three actual attempts; independently sampled goal unlike official reachable goals; state-level boat dynamics not paper's five visual planning simulators; source calibration uses repeated outcomes. No SOTA claim, no GPU scaling prediction. Stop tuning this development setup. Next legitimate comparison: official Table 16 with original frozen encoders/readouts/CEM, identical three attempts, disjoint calibration, same reachable goal and standardized distance threshold; original last residual vs ordinary GP vs strong fault identification/optimization controls. Evaluate all environments/conditions and three predictor seeds. Runtime DNS to GitHub and Hugging Face failed; official visual fits and generated data not accessible here. Local E24 archive holds code, protocol, results, profile and notebook. This note is findings only.