# Recursive guarded world model

Hypothesis: a discovered operation should be executable independently of its applicability, and later discoveries should change the interpretation of earlier observations.

The model is a recurrent convolutional executor, not a Transformer. Its initial copy/write vocabulary is supplied; the label dictionary, guards and conditional composition are learned. The active dictionary starts at four slots, splits high-error labels, and has a reserve of twelve. The same operations execute across three passes; label fields feed subsequent passes. Old training transitions get reassigned using the current dictionary. These are bounded, local experiments, not general program induction.

## Run

Open `colab.ipynb` in Google Colab and select a GPU (A100/L4 preferable; T4 supported in FP32). Or use:

```bash
git clone --depth 1 --branch research/recursive-world-model https://github.com/shekhawatxyz/fastai-2022
cd fastai-2022/experiments/recursive_world
python recursive_world.py --suite smoke --epochs 4 --seeds 1 --profile
python recursive_world.py --suite synthetic --epochs 12 --seeds 3 --ensemble --profile
```

For actual ARC human trajectories, use the released WARC implementation:

```bash
git clone https://github.com/third-intelligence/WARC /content/WARC
git -C /content/WARC checkout f0081ef00e867f63a08b1e4e7350d2baa0440267
pip install datasets einops timm huggingface_hub arc-agi==0.9.9 arcengine==0.9.3
python recursive_world.py --suite arc --games ls20 --sessions 1 --limit 24 --seeds 1 --arms direct,copy,small,stale,grow --profile --hard-arc --out arc_pilot
```

The default dataset is the public human-recordings mirror on Hugging Face. Alternatively extract the official ARC Prize Public Demo dataset and pass `--data /path/to/extracted`. Mirror byte identity has not been verified. The first qualifying recordings are selected deterministically, not by scores. Increase games, sessions, limit and seeds only after the pilot runs. Use held-out games for final conclusions, not hyperparameter tuning.

## Controls

| Arm | Purpose |
|---|---|
| direct | Direct color prediction |
| copy | Direct primitive selection, no learned dictionary |
| static | Same repeated neural computation, no intermediate execution/label feedback |
| recursive | Fixed 12-slot dictionary, recursive executor, no explicit retrospective target |
| fixed | 12-slot dictionary with retrospective relabeling |
| small | Four active slots, retrospective relabeling, no births |
| stale | Growing dictionary, initial pseudo-labels kept |
| grow | Growing dictionary plus retrospective relabeling |

All local models use the same width, recurrent passes, optimizer and input data (their heads have different parameter counts). Real replay uses three-frame context, 10 recent + 20 older transitions, official rotation transformations, 20 updates per observation by default, minibatch up to eight, gradient clipping, AdamW. WARC runs its released Muon/AdamW recipe. WARC frozen, pretrained TTT, and randomly initialized WARC TTT are separate comparisons. They differ in parameter count and pretraining; identical observations do not mean identical compute. Scratch WARC is the nearest pretraining control. BF16 is used where supported; unsupported GPUs run FP32, explicitly differing from published precision.

Optional ensembles average independently seeded predictions at additional cost; compare single models first. Hard execution is a separate score, not silently substituted for soft training.

## Outputs

- scores.csv: prequential changed-cell accuracy, false edits, exact frames, change F1, timing.
- epochs.csv: loss, gradient norm, active labels, cumulative update count, mean revision distance.
- weight_drift.csv: per-parameter distance from initialization, RMS update and cosine similarity.
- vocabulary.csv: active slots, primitive selection, confidence, entropy.
- births.csv: parent/child label events.
- paired_changed_accuracy.csv: recording-matched differences on ARC.
- trace.json / profile.txt: PyTorch CPU/CUDA profiler, a separate warmed-up update on the actual frame size.

## Falsification

The official MechEval scorer/annotations are not released. The published reference values in the script are **not comparable** to this all-changed-cell replay proxy. Evaluate frozen and TTT WARC on the same replay to make a valid comparison. No claim of novelty or benchmark superiority is made. To support the recursive mechanism, grow must beat stale, small and fixed, while controlling data, update count, exact-frame accuracy, false edits and wall time. If a direct copy baseline wins, reject the added machinery. If only the ensemble wins, attribute the gain to extra compute until proven otherwise. Synthetic experiments favor a supplied copy DSL; do not confuse them with independent evidence.

Development note: the preceding fixed-slot prototype was CPU-smoke-tested; the expanded code and full upstream integration still require execution in Colab. The script starts with assertions to detect obvious causal leakage, invalid hard execution, failed parameter updates and label growth errors.
