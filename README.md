# fast.ai 2022

My working repository for **Practical Deep Learning for Coders**, covering both parts of the 2022 course.

This repo is my source of truth for notebooks, exercises, experiments, notes, and any fixes needed to run the older course material on current environments.

## Structure

```text
.
├── part1/          # fastai/course22
├── part2/          # fastai/course22p2
├── experiments/    # my own models, variations, and side experiments
├── notes/          # lesson notes
└── README.md
```

## Workflow

- GitHub is the permanent source of truth.
- Colab, Kaggle, or local Jupyter are just execution environments.
- Course notebooks can be modified directly while following lessons.
- Larger deviations and original work go in `experiments/`.
- Commit meaningful progress regularly.
- Fix outdated dependencies as needed rather than trying to reproduce the exact 2022 environment forever.

## Upstream

Part 1 comes from:

`fastai/course22`

Part 2 comes from:

`fastai/course22p2`

The upstream course repositories are kept here as working material so I can modify the notebooks and preserve my own history.

# Progress

## Part 1

- [ ] 1: Getting started
- [ ] 2: Deployment
- [ ] 3: Neural net foundations
- [ ] 4: Natural Language (NLP)
- [ ] 5: From-scratch model
- [ ] 6: Random forests
- [ ] 7: Collaborative filtering
- [ ] 8: Convolutions (CNNs)

## Part 2

- [ ] 9: Stable Diffusion
- [ ] 9A: Deep Dive
- [ ] 9B: The Math of Diffusion
- [ ] 10: Diving Deeper
- [ ] 11: Matrix multiplication
- [ ] 12: Mean shift clustering
- [ ] 13: Backpropagation & MLP
- [ ] 14: Backpropagation
- [ ] 15: Autoencoders
- [ ] 16: The Learner framework
- [ ] 17: Initialization/normalization
- [ ] 18: Accelerated SGD & ResNets
- [ ] 19: DDPM and Dropout
- [ ] 20: Mixed Precision
- [ ] 21: DDIM
- [ ] 22: Karras et al (2022)
- [ ] 23: Super-resolution
- [ ] 24: Attention & transformers
- [ ] 25: Latent diffusion

## Experiments

Original experiments and extensions live in `experiments/`.

The point is not just to execute the notebooks. For each lesson, I should change things, make predictions before running cells, break implementations, rebuild pieces, and use the material for experiments I actually care about.

## Notes

Lesson notes live in `notes/`.

Keep these focused on:

- things I did not understand immediately
- implementation details worth remembering
- mistakes and debugging lessons
- ideas for experiments
- connections to papers or other projects
- questions worth returning to

## Goal

Work through the course deeply enough that I can reproduce the important ideas and implementations rather than merely recognize them when reading someone else's code.
