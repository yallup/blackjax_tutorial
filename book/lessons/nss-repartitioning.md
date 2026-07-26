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

# 03 · Nested sampling → posterior repartitioning

Lesson 2 gave us three reusable pieces: a Distrax prior, a log likelihood, and
a Pathfinder approximation. Now we will compose those same pieces in two new
ways:

1. run Nested Slice Sampling (NSS) for posterior samples **and** an evidence
   estimate;
2. turn Pathfinder into a proposal prior and use posterior repartitioning to
   reach the same answer with less compression.

The workflow follows the
[Nested Sampling Book's line-fitting tutorial](https://github.com/handley-lab/nested-sampling-book/blob/main/basic/line_fitting.ipynb)
and its
[posterior-repartitioning tutorial](https://github.com/handley-lab/nested-sampling-book/blob/main/advanced/supernest.ipynb),
using BlackJAX's
[NSS implementation](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/ns/nss/index.html)
throughout {cite:p}`yallup2026nss`.

## 1. Bring back the line

Each page in the book is an independent notebook, so this first cell recreates
the data and model from Lesson 2. Notice that nested sampling needs the prior
and likelihood separately; `log_posterior` alone is not enough.

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

prior = distrax.Joint(
    {
        "slope": distrax.Normal(loc=0.0, scale=2.0),
        "intercept": distrax.Normal(loc=0.0, scale=1.0),
        "log_noise": distrax.Normal(loc=-1.0, scale=0.75),
    }
)


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

## 2. What nested sampling adds

The Bayesian evidence is the average likelihood under the prior,

$$
\mathcal{Z} = \int \mathcal{L}(\theta)\,\pi(\theta)\,\mathrm{d}\theta.
$$

Nested sampling estimates this integral by maintaining a population of
**live points**. At every step it removes the lowest-likelihood points, records
them as **dead points**, and replaces them with points above the new likelihood
threshold {cite:p}`skilling2006nested`.

Three settings control the small run below:

- `num_live` controls resolution;
- `num_inner_steps` controls how thoroughly each replacement point moves;
- `num_delete` controls how many points are replaced together.

For NSS, use at least `max(5, 2 * dimension)` inner steps, then check that
increasing the value does not materially change the answer. Eight is a
conservative workshop value for our three-dimensional model.

```{code-cell} ipython3
num_live = 150
num_delete = 50
num_inner_steps = 8


def run_nss(key, logprior, loglike, initial_particles, max_batches=200):
    """Run NSS until the remaining live evidence is below five per cent."""
    sampler = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglike,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )

    key, init_key = jax.random.split(key)
    live = sampler.init(initial_particles, init_key)
    step = jax.jit(sampler.step)
    dead = []

    for batch in range(max_batches):
        if live.integrator.logZ_live - live.integrator.logZ < -3.0:
            break
        key, step_key = jax.random.split(key)
        live, info = step(step_key, live)
        dead.append(info)
    else:
        raise RuntimeError("NSS did not reach the stopping rule")

    complete_run = blackjax.ns.utils.finalise(live, dead)
    return key, complete_run, len(dead)
```

Draw the initial live points from the same prior object whose `log_prob` NSS
will evaluate. This is the same single-source-of-truth rule used in Lesson 2.

```{code-cell} ipython3
key, prior_key, nss_key = jax.random.split(key, 3)
initial_live = prior.sample(seed=prior_key, sample_shape=(num_live,))

key, direct_run, direct_batches = run_nss(
    nss_key,
    prior.log_prob,
    log_likelihood,
    initial_live,
)

print(f"direct NSS: {direct_batches} batches, "
      f"{direct_batches * num_delete} dead points")
```

:::{admonition} Do not use the final live points as posterior draws
:class: warning

The live population follows the current likelihood contour. It is not an
equally weighted posterior sample. `finalise` joins the dead-point history to
the remaining live points; `blackjax.ns.utils.sample` then applies the nested
sampling weights.
:::

## 3. Recover evidence and posterior draws

Prior-volume shrinkage is random, so we simulate it several times to obtain an
evidence estimate and uncertainty. We then resample the weighted run into
ordinary, equally weighted posterior draws.

```{code-cell} ipython3
key, direct_z_key, direct_sample_key = jax.random.split(key, 3)

direct_logz = jax.scipy.special.logsumexp(
    blackjax.ns.utils.log_weights(direct_z_key, direct_run, shape=128),
    axis=0,
)
direct_samples = blackjax.ns.utils.sample(
    direct_sample_key,
    direct_run,
    shape=2_000,
).position

print(
    f"direct NSS log evidence: "
    f"{direct_logz.mean():.3f} ± {direct_logz.std():.3f}"
)
```

## 4. Ask Pathfinder for a proposal

Nested-sampling cost is governed by **compression**: how far the posterior is
from the prior. Pathfinder already gives us a cheap approximation near the
posterior. We turn its marginal means and standard deviations into a
normalised, factorised `distrax.Joint` proposal.

```{code-cell} ipython3
key, pf_init_key, pf_key, pf_draw_key = jax.random.split(key, 4)
pf_initial = prior.sample(seed=pf_init_key)

pathfinder = blackjax.pathfinder(log_posterior)
pf_state, _ = pathfinder.init(pf_key, pf_initial, ftol=1e-5)
pf_draws, _ = pathfinder.sample(pf_draw_key, pf_state, 2_000)

parameter_names = ("slope", "intercept", "log_noise")
pathfinder_proposal = distrax.Joint(
    {
        name: distrax.Normal(
            loc=pf_draws[name].mean(),
            scale=2.0 * pf_draws[name].std() + 1e-6,
        )
        for name in parameter_names
    }
)
```

Doubling each marginal scale makes the proposal enclose the Pathfinder draws
rather than fit them too tightly.

:::{admonition} Why use a factorised proposal?
:class: note

`distrax.Joint` samples and scores the same named PyTree as the model, so no
packing or unpacking is required. We give up the correlations in Pathfinder's
Gaussian approximation, which can make the proposal less efficient on a
strongly correlated problem. Repartitioning remains exact: the density-ratio
correction accounts for any difference between the proposal and posterior.
Only efficiency changes. A dense multivariate proposal is a useful later
optimisation when its extra vector conversion is justified.
:::

## 5. Repartition without changing the answer

Posterior repartitioning replaces the original prior $\pi$ with a normalised
proposal $\tilde\pi$, and corrects the likelihood by the exact density ratio
{cite:p}`chen2019repartitioning`:

$$
\tilde{\mathcal{L}}(\theta)
=
\mathcal{L}(\theta)
\frac{\pi(\theta)}{\tilde\pi(\theta)}.
$$

The product is unchanged:

$$
\tilde{\mathcal{L}}(\theta)\tilde\pi(\theta)
=
\mathcal{L}(\theta)\pi(\theta).
$$

For safety, we use the SuperNest idea: mix 90% of the Pathfinder proposal with
10% of the original prior {cite:p}`petrosyan2023supernest`. The original-prior
component preserves the tails and protects against a missed region.

```{code-cell} ipython3
pathfinder_weight = 0.9


def repartitioned_logprior(params):
    return jnp.logaddexp(
        jnp.log(pathfinder_weight)
        + pathfinder_proposal.log_prob(params),
        jnp.log1p(-pathfinder_weight)
        + prior.log_prob(params),
    )


def repartitioned_loglikelihood(params):
    return (
        log_likelihood(params)
        + prior.log_prob(params)
        - repartitioned_logprior(params)
    )
```

Here is a useful unit test. The original and repartitioned log joint densities
should agree to numerical precision:

```{code-cell} ipython3
key, check_key = jax.random.split(key)
check_position = prior.sample(seed=check_key)

original_joint = log_posterior(check_position)
repartitioned_joint = (
    repartitioned_logprior(check_position)
    + repartitioned_loglikelihood(check_position)
)

print("log-joint difference:", repartitioned_joint - original_joint)
```

The new live points must also come from the new proposal prior. For this
workshop run, we make the 90/10 split explicit: 90% come from Pathfinder and
10% come from the original prior.

```{code-cell} ipython3
num_pathfinder = round(pathfinder_weight * num_live)
num_original = num_live - num_pathfinder

key, pf_prior_key, original_prior_key = jax.random.split(key, 3)
pathfinder_particles = pathfinder_proposal.sample(
    seed=pf_prior_key,
    sample_shape=(num_pathfinder,),
)
original_particles = prior.sample(
    seed=original_prior_key,
    sample_shape=(num_original,),
)
initial_repartitioned = jax.tree.map(
    lambda pf, original: jnp.concatenate([pf, original]),
    pathfinder_particles,
    original_particles,
)
```

:::{admonition} A simple, stratified mixture
:class: note

With 50 live points this gives 45 Pathfinder draws and 5 original-prior draws.
It makes the intended safety mixture visible and avoids drawing particles that
will immediately be discarded. This fixed allocation is a **stratified**
version of the mixture. If independent and identically distributed mixture
draws are required, draw a fresh Bernoulli component label for every particle
instead; the density and correction above do not change.
:::

NSS itself does not change. We simply compose it with the repartitioned prior,
corrected likelihood, and matching initial particles:

```{code-cell} ipython3
key, repartitioned_key = jax.random.split(key)

key, repartitioned_run, repartitioned_batches = run_nss(
    repartitioned_key,
    repartitioned_logprior,
    repartitioned_loglikelihood,
    initial_repartitioned,
)
```

## 6. Same inference, shorter run

```{code-cell} ipython3
key, repartitioned_z_key, repartitioned_sample_key = jax.random.split(
    key, 3
)

repartitioned_logz = jax.scipy.special.logsumexp(
    blackjax.ns.utils.log_weights(
        repartitioned_z_key,
        repartitioned_run,
        shape=128,
    ),
    axis=0,
)
repartitioned_samples = blackjax.ns.utils.sample(
    repartitioned_sample_key,
    repartitioned_run,
    shape=2_000,
).position

print(f"{'run':>18}  {'log evidence':>18}  {'batches':>8}")
print(
    f"{'direct':>18}  "
    f"{direct_logz.mean():>8.3f} ± {direct_logz.std():.3f}  "
    f"{direct_batches:>8}"
)
print(
    f"{'repartitioned':>18}  "
    f"{repartitioned_logz.mean():>8.3f} "
    f"± {repartitioned_logz.std():.3f}  "
    f"{repartitioned_batches:>8}"
)
print(
    f"\nrun-length reduction: "
    f"{direct_batches / repartitioned_batches:.1f}×"
)
```

Both evidence estimates target the original model. The numerical values need
not be identical, but they should agree within their nested-sampling
uncertainties. Repartitioning removes much of the prior-to-posterior
compression, so it uses fewer batches and has a smaller evidence uncertainty.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

axes[0].scatter(
    direct_samples["slope"],
    direct_samples["intercept"],
    s=8,
    alpha=0.16,
    label="direct NSS",
)
axes[0].scatter(
    repartitioned_samples["slope"],
    repartitioned_samples["intercept"],
    s=8,
    alpha=0.16,
    label="repartitioned NSS",
)
axes[0].scatter(
    pf_draws["slope"][::8],
    pf_draws["intercept"][::8],
    s=8,
    alpha=0.16,
    label="Pathfinder",
)
axes[0].set(
    xlabel="slope",
    ylabel="intercept",
    title="Three routes to the posterior",
)
axes[0].legend(frameon=False, fontsize=8)

axes[1].scatter(
    x,
    y,
    s=18,
    color="tab:blue",
    alpha=0.75,
    label="observations",
)
for i in range(0, 2_000, 40):
    line = (
        repartitioned_samples["slope"][i] * x
        + repartitioned_samples["intercept"][i]
    )
    axes[1].plot(x, line, color="tab:orange", alpha=0.08)
axes[1].set(
    xlabel="x",
    ylabel="y",
    title="Repartitioned NSS posterior lines",
)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()
```

## The composable-inference lesson

Nothing was rewritten from scratch:

- Distrax still owns prior sampling and prior density evaluation.
- The same pure `log_likelihood` works in Pathfinder and NSS.
- Pathfinder becomes a proposal, not a replacement for exact weighting.
- Posterior repartitioning changes the route through parameter space while
  preserving the original posterior and evidence.

That is composable inference: algorithms exchange small, testable functions and
PyTrees rather than taking ownership of the model.

:::{admonition} Checkpoint
:class: tip

You should now be able to explain why NSS needs separate prior and likelihood
functions, why final live points are not posterior samples, and why the density
ratio makes repartitioning exact.
:::

```{bibliography}
:filter: docname in docnames
```
