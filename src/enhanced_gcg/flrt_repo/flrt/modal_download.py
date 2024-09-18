import copy
import os

from flrt.modal_defs import default_params, hf_cache, stub

model_names = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-Guard-2-8B",
    "meta-llama/LlamaGuard-7b",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b",
    "google/gemma-7b-it",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "GraySwanAI/Llama-3-8B-Instruct-RR",
]


params = copy.copy(default_params)
params["gpu"] = None


@stub.function(**params)
def download_models():
    import transformers
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir()

    def is_model_already_downloaded(model_name):
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                return True
        else:
            return False

    for mn in model_names:
        if not is_model_already_downloaded(mn):
            print(f"Model {mn} is not cached. Downloading.")
            transformers.AutoTokenizer.from_pretrained(
                mn,
                token=os.environ.get("HUGGINGFACE_TOKEN", None),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            transformers.AutoModelForCausalLM.from_pretrained(
                mn, token=os.environ["HUGGINGFACE_TOKEN"], trust_remote_code=True
            )
            hf_cache.commit()
        else:
            print(f"Model {mn} is already cached.")


if __name__ == "__main__":
    with stub.run():
        download_models.remote()
