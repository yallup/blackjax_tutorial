# CosmicExplosions 2026 · BlackJAX tutorial — learner branch

This is the workshop branch. All three lessons are collected in one real
Jupyter notebook with deliberate gaps:

**`CosmicExplosions2026_BlackJAX_tutorial.ipynb`**

The complete, executed MyST lessons are on `main` and at
[yallup.github.io/blackjax_tutorial](https://yallup.github.io/blackjax_tutorial/).

## Start the workshop

Clone this branch:

```bash
git clone --branch learner --single-branch \
  git@github.com:yallup/blackjax_tutorial.git
cd blackjax_tutorial
```

Install the locked workshop environment:

```bash
uv sync
```

or install the project into your current Python environment:

```bash
python -m pip install .
```

Open `CosmicExplosions2026_BlackJAX_tutorial.ipynb`, select the environment
you just installed, and work through:

- **Lesson 1:** setup;
- **Lesson 2:** a Distrax line-fitting model and Pathfinder;
- **Lesson 3:** Nested Slice Sampling and Pathfinder-informed posterior
  repartitioning.

Search the notebook for `TODO` to find every exercise. The learner branch has
no website build or publishing dependencies; everything needed by the notebook
is declared in `pyproject.toml`.

## Sources

The tutorial is adapted from the
[BlackJAX Sampling Book](https://github.com/blackjax-devs/sampling-book) and
the [Nested Sampling Book](https://github.com/handley-lab/nested-sampling-book).
The final reading list connects this material to
[Nested Slice Sampling](https://arxiv.org/abs/2601.23252),
[posterior repartitioning](https://arxiv.org/abs/1908.04655), and
[SwiG](https://github.com/yallup/swig).
