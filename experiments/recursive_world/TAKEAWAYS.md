# Research takeaways — 2026-11-25

- E23: Standard GP residual memory reduced prediction MSE by 60.6% (0.014096535 to 0.005552974) versus last-residual reuse. The adaptive mixture lost to the standard GP. [Results](E23_DECISION.md)
- E24: Greedy GP planning reduced mean squared decision regret by 73.2% (0.001290939 to 0.000345482), while threshold success was essentially unchanged (39.9023% to 39.9219%). [Results](E24_DECISION.md)
- E22: Matrix residual transport reduced MSE by 22.1% in the turbulent-current regime (0.035662 to 0.027780), using extra paired calibration information. [Results](E22_DECISION.md)
- E16: Compositional Bayesian inference demonstrated backward reinterpretation of learned operations in a deliberately structured synthetic task. [Notebook](experiment_16_event_composition.ipynb)

Controlled numerical/synthetic results only, not official benchmark superiority or novelty claims. E25 is unscored.

## Rules for future research

- Verify novelty, strongest baseline, accessible benchmark, artifacts, and resource requirements before coding.
- Freeze a falsifiable hypothesis and run the cheapest decisive test first with matched information and compute.
- Scale only after a predefined gate passes; stop when the strongest simple baseline wins.
- Own the scientific judgment; use AI for implementation and adversarial review, not as a substitute for understanding.

Archive the exploratory line. No further GPU run to justify sunk costs.
