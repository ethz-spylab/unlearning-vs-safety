# An Adversarial Perspective on Machine Unlearning for AI Safety
This repository provides an original implementation of paper "An Adversarial Perspective on Machine Unlearning for AI Safety"

## Setup
There are several global variables and paths that we use throughout the project defined in `src\util\globals.py`. To adapt the code for your needs provide appropriate information there.

For the majority of programs you should be fine using default conda environment `unlearning` that you can install via:
```
conda env create -f env.yml
```

If there is another `env.yml` present in the subfolder of interest that part of the project requires it. You can install it the same way as described above, but with correct path to the file.

Lastly, we have extensively used Weights and Biases, and HuggingFace, so to use our code without any issues you should set all access tokens and relevant environment variables such as `HF_DATASETS_CACHE` or `HF_DATASETS_CACHE`.

## Running experiments
The subfolders in `src\` correspond to a specific method of knowledge extraction or unlearning method listed in the paper. 

This repository has been configured as a python module so to run it you need to add flag `-m` for instance like this:
```
CUDA_VISIBLE_DEVICES=2 python -m src.benchmarking.benchmark --model_name_or_path cais/Zephyr_RMU
```
Furthermore, you need to run it from the highest repository level i.e. the one that contains `src\` folder.

Ultimately, directory `scripts\` contains scripts used for all experiments in this paper. You should be able to simple use them in the following way:
```
bash scripts/benchmark_model.sh
```

