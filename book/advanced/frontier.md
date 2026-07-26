# Beyond the workshop

Stop here for the core tutorial. The material below is a reading map, not an
extra exercise.

## From posterior approximation to nested sampling

Lesson 3 used Pathfinder and Nested Slice Sampling together on a small model.
Nested sampling also estimates the Bayesian evidence needed for model
comparison. The
[Nested Sampling Book](https://github.com/handley-lab/nested-sampling-book) is
the natural next stop after this workshop.

### Nested Slice Sampling

[Nested Slice Sampling: Vectorized Nested Sampling for GPU-Accelerated
Inference](https://arxiv.org/abs/2601.23252) introduces a GPU-friendly,
vectorised nested-sampling formulation with Hit-and-Run slice updates
{cite:p}`yallup2026nss`. The paper reports accurate evidence and posterior
estimates on high-dimensional and multimodal targets; the open-source
implementation lives in BlackJAX.

Read this when you want to understand:

- why nested sampling is difficult to parallelise;
- how constrained slice updates fit the BlackJAX design;
- when evidence estimation changes the inference question.

### SwiG for hierarchical models

[Nested Sampling with Slice-within-Gibbs](https://arxiv.org/abs/2602.17414)
targets hierarchical models whose likelihood can be decomposed into blocks
{cite:p}`yallup2026swig`. Its budget decomposition avoids recomputing the whole
likelihood for every local update.

The research implementation is
[yallup/swig](https://github.com/yallup/swig). It is intentionally advertised
here as an advanced destination rather than a workshop dependency.

```bash
git clone git@github.com:yallup/swig.git
cd swig
uv sync --extra examples
```

## A sensible progression

1. Keep a well-factored model: named PyTrees, a prior object, and pure density
   functions.
2. Use Pathfinder for a rapid approximation and diagnostic.
3. Use BlackJAX nested sampling when evidence is part of the question.
4. Investigate SwiG when the model is genuinely hierarchical and block
   structure can be exploited.

```{bibliography}
:filter: docname in docnames
```
