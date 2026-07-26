# CosmicExplosions 2026 · BlackJAX tutorial

A short MyST tutorial that moves from setup to a structured Bayesian model,
Pathfinder, Nested Slice Sampling, and posterior repartitioning in BlackJAX.

## Start the workshop

Clone the learner branch, which contains deliberate gaps:

```bash
git clone --branch learner --single-branch \
  git@github.com:yallup/blackjax_tutorial.git
cd blackjax_tutorial
```

Install the locked workshop environment:

```bash
uv sync
```

Then open `CosmicExplosions2026_BlackJAX_tutorial.ipynb` and search for `TODO`.
The complete answers live on `main`. If `uv` is unavailable, run
`python -m pip install .` instead.

## Build

```bash
make build
```

The static site is written to `book/_build/html/`. A GitHub Actions workflow
publishes that directory to GitHub Pages whenever `main` changes.

## Sources

The tutorial is adapted from the
[BlackJAX Sampling Book](https://github.com/blackjax-devs/sampling-book) and
the [Nested Sampling Book](https://github.com/handley-lab/nested-sampling-book).
The final reading list connects this material to
[Nested Slice Sampling](https://arxiv.org/abs/2601.23252),
[posterior repartitioning](https://arxiv.org/abs/1908.04655), and
[SwiG](https://github.com/yallup/swig).
