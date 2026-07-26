# [CosmicExplosions 2026](https://cambridgetransients.github.io/CosmicExplosions2026/) · BlackJAX tutorial

Build a small Bayesian workflow from the ground up: a clean Python environment,
a model whose assumptions are explicit, a fast Pathfinder approximation, and
a nested-sampling run that can be accelerated through posterior repartitioning.

This is a short, practical route for a general audience. You need basic Python,
but you do not need prior experience with JAX or Bayesian computation.

:::{admonition} Start on the learner branch
:class: tip

The website shows the complete route. To follow along with the exercises, clone
the branch with deliberate gaps:

```bash
git clone --branch learner --single-branch \
  git@github.com:yallup/blackjax_tutorial.git
cd blackjax_tutorial
uv sync
uv run jupyter notebook CosmicExplosions2026_BlackJAX_tutorial.ipynb
```

The last command starts Jupyter and opens the learner notebook in your browser.

Prefer HTTPS? Replace the clone URL with
`https://github.com/yallup/blackjax_tutorial.git`.
:::

## The route

| Stop | What you will do | What you will leave with |
| --- | --- | --- |
| **01 · Setup** | Create a reproducible environment and register its kernel | A working BlackJAX notebook kernel |
| **02 · Model → Pathfinder** | Express a structured model with Distrax, then approximate it | Posterior draws and a reusable log density |
| **03 · NSS → Repartitioning** | Run Nested Slice Sampling, then reuse Pathfinder as a proposal | Posterior draws, evidence, and a shorter repartitioned run |
| **Beyond** | Read the research frontier | A map from NSS to SwiG |

The teaching structure follows the
[BlackJAX Sampling Book](https://github.com/blackjax-devs/sampling-book), while
the modelling recommendations and advanced path draw on the
[Nested Sampling Book](https://github.com/handley-lab/nested-sampling-book).

```{tableofcontents}
```
