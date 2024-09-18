import os

import modal

hf_cache_path = "/root/model_store"
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
caiplay_path = "/root/output"
s3_access_credentials = modal.Secret.from_name("s3-access")
caiplay_bucket = modal.CloudBucketMount("caiplay", secret=s3_access_credentials)
logs_path = "/root/wandb"
logs_volume = modal.Volume.from_name("wandb-logs", create_if_missing=True)

gpu = {
    "A10G": modal.gpu.A10G(),
    "A100": modal.gpu.A100(size="40GB"),
    "A100-80": modal.gpu.A100(size="80GB"),
    "H100": modal.gpu.H100(),
}[os.environ.get("MODAL_GPU", "H100")]

default_params = dict(
    retries=0,
    timeout=60 * 60 * 24,
    cpu=8,
    gpu=gpu,
    memory=64 * 1024,
    secrets=[
        modal.Secret.from_name("s3-access"),
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("openai"),
    ],
    mounts=[
        modal.Mount.from_local_dir("data", remote_path="/root/data"),
    ],
    concurrency_limit=int(os.environ.get("MODAL_CONCURRENCY_LIMIT", "10")),
    volumes={
        logs_path: logs_volume,
        caiplay_path: caiplay_bucket,
        hf_cache_path: hf_cache,
    },
)

packages = [
    "numpy",
    "transformers>=4.40.0",
    "typer>=0.9",
    "accelerate",
    "tqdm",
    "boto3",
    "ninja",
    "packaging",
    "pandas",
    "s3fs",
    "mosaicml-streaming",
    "datasets",
    "wandb>=0.17.2",
    "peft",
    "trl",
    "instructor",
    "pydantic",
]
image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
    .run_commands(
        "apt-get update",
        "apt-get install -y git wget",
        "uv pip install --system --compile-bytecode " + " ".join(packages),
        "pip3 install flash-attn --no-build-isolation",
    )
    .env(dict(HF_HOME=hf_cache_path))
)
stub = modal.App("flrt", image=image)
