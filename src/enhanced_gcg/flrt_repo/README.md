# This repo is based on Fluent student-teacher redteaming
It includes several new losses and penalties. Furthermore it allows you to use any model as a target and not only its finetuned version. All you need is in `demo.py`

## Fluent dreaming for language models.

The code here implements the discrete prompt optimization algorithms in the paper ["Fluent student-teacher redteaming"](https://arxiv.org/abs/2407.17447).

[Please also see the companion page that demonstrates using the code here.](https://confirmlabs.org/posts/flrt.html)

The `demo.ipynb` file here is the source for that companion page.

Key modules:
- `flrt.attack`: The main attack entrypoint including the AttackConfig object.
- `flrt.victim`: Code for managing attack "victims" - the model that will be forced to misbehave.
- `flrt.templates`: Attack templates specifying which subset of the prompt can be optimized by the discrete optimization.
- `flrt.util`: Tools for loading models and tokenizers and generating.

The remaining code is either internal to the algorithm (`flrt.objective`, `flrt.operators`) or is scaffolding for running on Modal (`flrt.modal_defs`, `flrt.modal_download`) or running evaluations (`flrt.judge`).
