# BlackJAX: a field guide to inference — learner branch

This is the workshop branch. It contains deliberate gaps in Lesson 2 for you to
fill. The complete answers are on `main`.

## Start the workshop

Clone this branch:

```bash
git clone --branch learner --single-branch \
  git@github.com:yallup/blackjax_tutorial.git
cd blackjax_tutorial
```

Install the project and book dependencies with `uv`:

```bash
uv sync --group book
```

Then preview the unexecuted book:

```bash
make preview
```

Open `book/lessons/model-pathfinder.md` and search for `TODO`. If `uv` is
unavailable, the setup lesson includes a `pip install` fallback.

## Check your answers

```bash
make check
```

`make check` executes every cell. A successful build is your confirmation that
the model and Pathfinder composition are complete.

## Sources

The tutorial is adapted from the
[BlackJAX Sampling Book](https://github.com/blackjax-devs/sampling-book) and
the [Nested Sampling Book](https://github.com/handley-lab/nested-sampling-book).
The final reading list connects this material to
[Nested Slice Sampling](https://arxiv.org/abs/2601.23252) and
[SwiG](https://github.com/yallup/swig).
