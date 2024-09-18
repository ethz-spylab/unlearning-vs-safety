import dataclasses
import os
from typing import List, Optional

import torch
import torch.distributions
import transformers


def mellowmax(t: torch.Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device="cpu"))
        )
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
    model_name: str
    peft_path: str
    response_template: str
    first_token: str
    system_prompt: str
    sep: str
    # If banned_tokens is None, then banned tokens will be set to:
    # tokenizer.all_special_ids + range(tokenzier.vocab_size, num_embeddings)
    banned_tokens: List[int] = None
    tokenizer_name: Optional[str] = None

    @property
    def short_name(self):
        return self.model_name.split("/")[-1]


ft_datetime = "240506_223106"
model_configs = {
    "cais/Zephyr_RMU": ModelConfig(
        model_name="cais/Zephyr_RMU",
        peft_path=None,
        response_template="<|assistant|>\n",
        first_token="",
        system_prompt=None,
        sep=" ",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    ),
    "J4Q8/zephyr-npo-bio": ModelConfig(
        model_name="J4Q8/zephyr-npo-bio",
        peft_path=None,
        response_template="<|assistant|>\n",
        first_token="",
        system_prompt=None,
        sep=" ",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    ),
    "J4Q8/zephyr-dpo-bio": ModelConfig(
        model_name="J4Q8/zephyr-dpo-bio",
        peft_path=None,
        response_template="<|assistant|>\n",
        first_token="",
        system_prompt=None,
        sep=" ",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    ),
    "J4Q8/zephyr-npo-cyber": ModelConfig(
        model_name="J4Q8/zephyr-npo-bio",
        peft_path=None,
        response_template="<|assistant|>\n",
        first_token="",
        system_prompt=None,
        sep=" ",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    ),
    "J4Q8/zephyr-dpo-cyber": ModelConfig(
        model_name="J4Q8/zephyr-dpo-bio",
        peft_path=None,
        response_template="<|assistant|>\n",
        first_token="",
        system_prompt=None,
        sep=" ",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    ),
    "llama-2": ModelConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        peft_path=f"Llama-2-7b-chat-hf-{ft_datetime}",
        response_template="[/INST]",
        first_token="Here",
        system_prompt=None,
        sep=" ",
    ),
    "llama-3": ModelConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        peft_path="earlysoft-Meta-Llama-3-8B-Instruct-240722_214056",
        response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        first_token="Here",
        system_prompt=None,
        sep="",
    ),
    "rr": ModelConfig(
        model_name="GraySwanAI/Llama-3-8B-Instruct-RR",
        peft_path="earlysoft-Llama-3-8B-Instruct-RR-240711_152536",
        response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        first_token="Here",
        system_prompt=None,
        sep="",
    ),
    "phi-3": ModelConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        peft_path=f"Phi-3-mini-4k-instruct-{ft_datetime}",
        response_template="<|assistant|>\n",
        first_token="Here",
        system_prompt=None,
        sep=" ",
    ),
    "gemma-7b": ModelConfig(
        model_name="google/gemma-7b-it",
        peft_path=f"gemma-7b-it-{ft_datetime}",
        response_template="<end_of_turn>\n<start_of_turn>model\n",
        first_token="Sure",
        system_prompt=None,
        sep="",
    ),
    "gemma-2b": ModelConfig(
        model_name="google/gemma-2b-it",
        peft_path="fgemma-2b-it-240510_221755",
        response_template="<end_of_turn>\n<start_of_turn>model\n",
        first_token="Sure",
        system_prompt=None,
        sep="",
    ),
    "vicuna": ModelConfig(
        model_name="lmsys/vicuna-7b-v1.5",
        peft_path=f"vicuna-7b-v1.5-{ft_datetime}",
        response_template="ASSISTANT:",
        first_token="Here",
        system_prompt=None,
        sep=" ",
    ),
    "pythia-410m": ModelConfig(
        model_name="EleutherAI/pythia-410m",
        peft_path=None,
        response_template="",
        first_token="Here",
        system_prompt=None,
        sep="",
        banned_tokens=[0],
    ),
}


TDC_LLAMA_PROMPT = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] """

LLAMA_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def load_tokenizer(tokenizer_name):
    hf_login()

    revision = None
    if tokenizer_name == "lmsys/vicuna-7b-v1.5":
        # this revision of the vicuna tokenizer includes a chat template. it was later reverted
        # https://huggingface.co/lmsys/vicuna-7b-v1.5/commit/3321f76e3f527bd14065daf69dad9344000a201d
        # https://huggingface.co/lmsys/vicuna-7b-v1.5/discussions/16
        revision = "90cea6508421a940cb03bec1a7c18fc2abc07d63"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, revision=revision
    )
    tokenizer.padding_side = "left"
    if "llama-2" in tokenizer_name or "llama-3" in tokenizer_name:
        tokenizer.pad_token = tokenizer.unk_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def hf_login():
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)

    if hf_token is not None and not hf_login.done:
        from huggingface_hub import login

        login(token=hf_token)
        hf_login.done = True


hf_login.done = False


def load_model(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=None,
    requires_grad=False,
):
    if "gpt2" in model_name:
        attn_implementation = "eager"
    elif attn_implementation is None:
        attn_implementation = "flash_attention_2"

    hf_login()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    if not requires_grad:
        # If we're not optimizing any model parameters, mark them
        # requires_grad(False). This will dramatically reduce memory
        # requirements.
        model.requires_grad_(False)
    return model


def load_model_and_tokenizer(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=None,
    tokenizer_name=None,
    requires_grad=False,
):
    hf_login()
    model = load_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        requires_grad=requires_grad,
    )
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path
    tokenizer = load_tokenizer(tokenizer_name)
    return model, tokenizer


def hf_generate(model, input_ids, attention_mask=None, **kwargs):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    settings = dict(
        max_new_tokens=512,
        num_return_sequences=1,
        temperature=1.0,  # needed to get rid of warning?!
        top_p=1.0,  # needed to get rid of warning?!
        do_sample=False,  # argmax sampling, ignores the temp/top_p args
    )
    settings.update(kwargs)
    return model.generate(input_ids, attention_mask=attention_mask, **settings)


def load_vllm(model_name, tokenizer_name=None):
    hf_login()
    from vllm import LLM

    model = LLM(model=model_name)
    return model, load_tokenizer(model_name)


def vllm_generate(
    model, prompts=None, prompt_token_ids=None, use_tqdm=True, max_tokens=200, **kwargs
):
    from vllm import SamplingParams

    settings = {
        "temperature": 0,
        "n": 1,
        "max_tokens": max_tokens,
    }
    settings.update(kwargs)
    params = SamplingParams(**settings)
    outputs = model.generate(
        prompts=prompts,
        prompt_token_ids=prompt_token_ids,
        sampling_params=params,
        use_tqdm=use_tqdm,
    )
    return [o.outputs[0].text for o in outputs], outputs


def memory_usage():
    print(torch.cuda.memory_summary())
    import gc

    pt_objs = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if obj.nelement() > (1000 * 1000):
                    pt_objs.append(obj)

        except:  # noqa
            pass
    for obj in pt_objs:
        print(type(obj), obj.size(), obj.dtype, obj.numel() * obj.element_size() / 1e9)
