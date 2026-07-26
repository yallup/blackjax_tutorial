---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 01 · Setup

The goal of this lesson is deliberately modest: make one environment that the
terminal, Jupyter, and your editor all agree on.

## Clone the exercises

```bash
git clone --branch learner --single-branch \
  git@github.com:yallup/blackjax_tutorial.git
cd blackjax_tutorial
```

The `learner` branch contains the gaps. The complete lesson is on `main`.

## Recommended: sync with uv

The repository includes a `pyproject.toml` and a locked dependency set. From
the project directory, run:

```bash
uv sync --group book
```

This creates `.venv/` and installs BlackJAX, Distrax, NumPy, Matplotlib,
IPython, the notebook kernel, and the MyST book tools. You do not need to
activate the environment when you prefix commands with `uv run`.

Register the environment as a notebook kernel:

```bash
uv run python -m ipykernel install --user \
  --name blackjax-tutorial \
  --display-name "Python (BlackJAX tutorial)"
```

:::{admonition} No uv?
:class: note

A plain `pip` environment is fine for following the two lessons:

```bash
python -m pip install blackjax distrax numpy matplotlib ipython ipykernel
python -m ipykernel install --user \
  --name blackjax-tutorial \
  --display-name "Python (BlackJAX tutorial)"
```
:::

## Smoke test

Run this cell using **Python (BlackJAX tutorial)**:

```{code-cell} ipython3
import blackjax
import jax
import matplotlib
import numpy as np

print("BlackJAX:", blackjax.__version__)
print("JAX device:", jax.devices()[0])
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
```

You are ready when the cell prints four lines without an import error. A CPU
device is exactly right for this tutorial.

## A tiny JAX check

JAX works with explicit random keys. Splitting a key makes the source of each
random draw visible and reproducible:

```{code-cell} ipython3
key = jax.random.key(7)
key, draw_key = jax.random.split(key)
draws = jax.random.normal(draw_key, shape=(5,))
draws
```

:::{admonition} Checkpoint
:class: tip

Keep the kernel selected and move on. Lesson 2 will reuse this environment,
this explicit-key pattern, and the idea that every piece of the model should be
inspectable.
:::
