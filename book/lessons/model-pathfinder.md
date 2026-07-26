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

# 02 · Model → Pathfinder

Inference is easier to change, test, and reuse when the model and the algorithm
are separate. We will first build a small line-fitting model in
[Distrax](https://github.com/google-deepmind/distrax), then hand its log density
to BlackJAX Pathfinder.

This follows the best-practice route in the
[Nested Sampling Book's line-fitting lesson](https://github.com/handley-lab/nested-sampling-book/blob/main/basic/line_fitting.ipynb):
use named parameters, keep sampling and log-density evaluation on the same
prior object, and make the likelihood a readable function.

## 1. Make some data

We will infer the slope, intercept, and noise of a straight line.

```{code-cell} ipython3
import blackjax
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

key = jax.random.key(17)
x = jnp.linspace(-2.0, 2.0, 40)
key, data_key = jax.random.split(key)
y = 0.8 * x - 0.2 + 0.25 * jax.random.normal(data_key, shape=x.shape)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(x, y, s=24, color="tab:blue", label="observations")
ax.set(xlabel="x", ylabel="y", title="A noisy straight line")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
plt.show()
```

## 2. Put the prior in one place

A dictionary is a JAX *PyTree*: a named structure that JAX and BlackJAX can
transform, differentiate, and vectorise. Names such as `slope` are much harder
to misuse than positions such as `params[0]`.

Distrax's `Joint` distribution gives us both `sample` and `log_prob`. That
single source of truth prevents the prior we draw from drifting away from the
prior density used during inference.

```{code-cell} ipython3
prior = distrax.Joint(
    {
        "slope": distrax.Normal(loc=0.0, scale=2.0),
        "intercept": distrax.Normal(loc=0.0, scale=1.0),
        "log_noise": distrax.Normal(loc=-1.0, scale=0.75),
    }
)
```

Keep the likelihood separate and write its Gaussian log density directly in
JAX. This makes the numerical model explicit and easy to test, replace, or
reuse in another inference algorithm:

```{code-cell} ipython3
def log_likelihood(params):
    noise = jnp.exp(params["log_noise"])
    mean = params["slope"] * x + params["intercept"]
    standardised_residual = (y - mean) / noise
    return -0.5 * jnp.sum(
        standardised_residual**2
        + 2.0 * jnp.log(noise)
        + jnp.log(2.0 * jnp.pi)
    )


def log_posterior(params):
    return prior.log_prob(params) + log_likelihood(params)
```

Before inference, check that a prior draw has the shape and names you expect:

```{code-cell} ipython3
key, init_key = jax.random.split(key)
initial_position = prior.sample(seed=init_key)

initial_position, log_posterior(initial_position)
```

:::{admonition} Best-practice pattern
:class: tip

1. Use semantic parameter names in a PyTree.
2. Let one distribution object own both prior sampling and `log_prob`.
3. Keep `log_likelihood` and `log_posterior` as small, pure functions.
4. Test one draw before compiling or sampling.
:::

## 3. PPLs and the BlackJAX philosophy

Distrax is a distribution and bijector library rather than a complete
probabilistic programming language (PPL). It is a good fit here because our
model is small and writing the likelihood explicitly makes every assumption
visible. [TensorFlow Probability](https://www.tensorflow.org/probability) is
another option: it provides a much larger collection of distributions,
bijectors, joint distributions, and inference tools, including a
[JAX substrate](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions).

Both libraries can compose named distributions:
[Distrax `Joint`](https://github.com/google-deepmind/distrax#joint-distributions)
is what we use above, while TFP provides `JointDistribution`. These objects
bundle sampling and joint-density evaluation, but that capability alone does
not make either one a PPL.

Full PPLs such as NumPyro, PyMC, and Stan go further. They let you describe a
generative model and typically manage tasks such as parameter transforms,
conditioning, naming latent variables, and assembling the joint log density.
That is extremely useful as models grow.

BlackJAX deliberately sits at a different layer:

| Layer | Main question | Examples |
| --- | --- | --- |
| Distribution toolkit | How do I sample from, score, and compose distributions? | Distrax (`Joint`), TensorFlow Probability (`JointDistribution`) |
| Probabilistic programming | How do I express and manage a generative model? | NumPyro, PyMC, Stan |
| Inference algorithms | How does state move through this target density? | BlackJAX Pathfinder, NUTS, SMC, NSS |

BlackJAX is therefore an **inference library, not a modelling language**. Its
[design principles](https://blackjax-devs.github.io/blackjax/developer/design_principles.html)
favour small pure functions, explicit random keys and state, PyTrees, and
`init`/`step` interfaces. The trade-off is visible plumbing: we supply a log
density—or, for nested sampling, separate prior and likelihood functions.
The payoff is that the model is not owned by one algorithm, and algorithmic
pieces can be inspected, replaced, transformed with JAX, and recomposed.

These approaches are complementary. A larger project can define its model in a
PPL, extract a JAX-compatible log-density function, and pass that function to
BlackJAX. The BlackJAX documentation demonstrates this with
[NumPyro](https://blackjax-devs.github.io/blackjax/examples/howto_use_numpyro.html),
[PyMC](https://blackjax-devs.github.io/blackjax/examples/howto_use_pymc.html),
and
[TensorFlow Probability](https://blackjax-devs.github.io/blackjax/examples/howto_use_tfp.html).
We use Distrax today to keep that boundary easy to see; TFP is an equally valid
model-building route, but it is not an extra workshop dependency.

## 4. Compose the model with Pathfinder

[Pathfinder](https://arxiv.org/abs/2108.03782) follows an L-BFGS optimisation
path and builds local Gaussian approximations from the optimiser's inverse
Hessian estimates {cite:p}`zhang2022pathfinder`. BlackJAX exposes it with an
`init` phase for the expensive approximation and a `sample` phase for cheap
draws, matching the composable interface used throughout the
[Sampling Book Pathfinder lesson](https://blackjax-devs.github.io/sampling-book/algorithms/pathfinder.html).

```{code-cell} ipython3
pathfinder = blackjax.pathfinder(log_posterior)

key, fit_key, sample_key = jax.random.split(key, 3)
state, info = pathfinder.init(
    fit_key,
    initial_position,
    ftol=1e-5,
)
samples, _ = pathfinder.sample(sample_key, state, 2_000)
```

The model did not need to know which algorithm would consume it. Pathfinder
needed only a log density and an initial PyTree.

```{code-cell} ipython3
for name in ("slope", "intercept", "log_noise"):
    values = samples[name]
    print(
        f"{name:>10}: "
        f"mean={values.mean(): .3f}, "
        f"sd={values.std(): .3f}"
    )
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

axes[0].scatter(x, y, s=18, color="tab:blue", alpha=0.75)
for i in range(0, 2_000, 40):
    line = samples["slope"][i] * x + samples["intercept"][i]
    axes[0].plot(x, line, color="tab:orange", alpha=0.08)
axes[0].set(xlabel="x", ylabel="y", title="Posterior lines")

axes[1].scatter(
    samples["slope"],
    samples["intercept"],
    s=8,
    alpha=0.2,
    color="tab:green",
)
axes[1].set(xlabel="slope", ylabel="intercept", title="Pathfinder draws")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()
```

## Why “composable” matters

The useful object is not a one-off script; it is `log_posterior`. The same
function can be passed to HMC, NUTS, SMC, nested sampling, or another
variational method. Pathfinder can be the approximation you report, a rapid
diagnostic, or an informed starting point for a later sampler.

For example, BlackJAX also provides `pathfinder_adaptation`, which uses the
Pathfinder inverse-Hessian estimate to initialise an MCMC kernel:

```{code-block} python
adapt = blackjax.pathfinder_adaptation(blackjax.nuts, log_posterior)
(warm_state, parameters), _ = adapt.run(
    warmup_key, initial_position, 400
)
nuts = blackjax.nuts(log_posterior, **parameters)
```

That extension is intentionally not part of today's exercise. The important
move is already complete: **model first, inference second**.

:::{admonition} Checkpoint
:class: tip

You should now be able to explain where the prior lives, where the likelihood
lives, and exactly what Pathfinder receives.
:::

```{bibliography}
:filter: docname in docnames
```
