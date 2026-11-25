# E25: selective disturbance transfer — executable bundle

Frozen hypothesis: [E25_BET.md](../E25_BET.md). Upstream pinned commit 5c84bba92b2ff19d41175070a61238fa1396d10d.
This directory supersedes the incomplete local pilot notebook. Nothing is run in GitHub Actions.

- e25.py: frozen predictor correction and controls.
- run_e25.py: disjoint calibration and official episodes.
- run_suite.py: complete reproduction, independent baseline, calibration, evaluation, and scoring; required explicit pilot/full scope.
- score_e25.py: rejects missing/duplicate records; parity against independently executed upstream. Pilot cannot confirm the gate.
- test_e25.py: CPU unit tests.
- E25_Colab.ipynb: self-contained setup and both scopes with automatic scoring.
- tests.txt: actual local unit test output.

Run in an environment with upstream's pinned requirements, Python 3.11, compatible CUDA driver R580+ and EGL. In the notebook, select GPU and set SCOPE to pilot (1 cell, 1800 test rows) or full (24 cells, 43200 rows). Full is resource-intensive. Do not change upstream sizes or tune on test results. On a local machine:
```bash
python run_suite.py --upstream /path/to/counterfactual-WM --work /path/to/results --scope pilot
python run_suite.py --upstream /path/to/counterfactual-WM --work /path/to/results --scope full
```
Run full only after the pilot and with sufficient storage. Both commands score automatically. A baseline parity failure invalidates the comparison. No upstream fit or GPU run has succeeded locally. No ACTUS reproduction, no SOTA result. Posterior mean, not full marginalization; no rotation or explicit fresh innovation estimator. The source is now present rather than only notes; previous experimental archives have not all been uploaded here.
