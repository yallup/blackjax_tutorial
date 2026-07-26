# BlackJAX: a field guide to inference

A short MyST/Jupyter Book tutorial that moves from a clean Python setup to a
structured Bayesian model in Distrax and a Pathfinder approximation in
BlackJAX.

## Start the workshop

Clone the learner branch, which contains deliberate gaps:

```bash
git clone --branch learner --single-branch \
  git@github.com:yallup/blackjax_tutorial.git
cd blackjax_tutorial
```

Install the project and book dependencies with `uv`:

```bash
uv sync --group book
```

Then preview the book:

```bash
make preview
```

The complete answers live on `main`. If `uv` is unavailable, the setup lesson
includes a `pip install` fallback.

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
[Nested Slice Sampling](https://arxiv.org/abs/2601.23252) and
[SwiG](https://github.com/yallup/swig).
