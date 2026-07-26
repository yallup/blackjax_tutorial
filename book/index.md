# BlackJAX: a field guide to inference

Build a small Bayesian workflow from the ground up: a clean Python environment,
a model whose assumptions are explicit, and a fast Pathfinder approximation
that can be composed with the wider BlackJAX toolbox.

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
uv sync --group book
```

Prefer HTTPS? Replace the clone URL with
`https://github.com/yallup/blackjax_tutorial.git`.
:::

## The route

| Stop | What you will do | What you will leave with |
| --- | --- | --- |
| **01 · Setup** | Create a reproducible environment and register its kernel | A working BlackJAX notebook kernel |
| **02 · Model → Pathfinder** | Express a structured model with Distrax, then approximate it | Posterior draws and a reusable log density |
| **Beyond** | Read the nested-sampling frontier | A map to Nested Slice Sampling and SwiG |

The teaching structure follows the
[BlackJAX Sampling Book](https://github.com/blackjax-devs/sampling-book), while
the modelling recommendations and advanced path draw on the
[Nested Sampling Book](https://github.com/handley-lab/nested-sampling-book).

```{tableofcontents}
```
