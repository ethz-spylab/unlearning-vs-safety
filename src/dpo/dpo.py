# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapted from: https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py"""

"""
# regular:
python examples/scripts/dpo.py \
    --dataset_proportions=0.5,0.25,0.25 \
    --model_name_or_path=HuggingFaceH4/zephyr-7b-beta \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpo.py \
    --dataset_proportions=0.5,0.25,0.25 \
    --model_name_or_path=HuggingFaceH4/zephyr-7b-beta \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import logging
import multiprocessing
import os
import datetime
import random
from copy import copy
from contextlib import nullcontext
from typing import Literal
from accelerate import Accelerator
from transformers import DataCollatorWithPadding

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from trl.env_utils import strtobool


from src.benchmarking.benchmark import Benchmark
from src.util.helpers import create_if_not_exists, save_as_json
from src.util.globals import OUTPUT_DIR, HF_USERNAME

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from dataclasses import dataclass, field
from datasets import load_dataset, interleave_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

def get_open_assistant_dataset():
    ds = load_dataset("javirandor/oasst2_dpo", split="train")
    ds = ds.rename_columns({"chosen_response": "chosen", "rejected_response": "rejected"})
    
    def sample2prompt(sample):
        return {"prompt": sample["prompt"][0]["content"],
                "chosen": sample["chosen"]["content"],
                "rejected": sample["rejected"]["content"]}
    
    ds = ds.map(sample2prompt)
    return ds

def mix_datasets(proportions: list[float], subset: Literal["bio","cyber"]):
    wmdp_forget = load_dataset(f"J4Q8/{subset}_forget_dpo", split="train")
    mc_retain = load_dataset(f"J4Q8/mc_{subset}_retain_dpo", split="train")
    openassistant = get_open_assistant_dataset()
    return interleave_datasets([wmdp_forget, mc_retain, openassistant], proportions, seed=42)


@dataclass
class ExperimentConfig:
    dataset_proportions: str = field(
        default="0.5,0.25,0.25", metadata={"help": "coma separated proportions for wmdp-forget, arc-mc-retain, openassistant"}
    )
    subset: Literal["bio", "cyber"] = field(default="bio", metadata={"help": "bio or cyber subset for refusal training"})
    use_chat_template_p: float = field(default=0.5, metadata={"help": "probability of using applying the chat template to each sample"})


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig, ExperimentConfig))
    args, training_args, model_config, exp_args= parser.parse_args_and_config()
       
    
    # make sure the output directory exists
    date = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    formatted_model_name = model_config.model_name_or_path.replace("/", "-")
    training_args.output_dir = os.path.join(OUTPUT_DIR, f"dpo-{formatted_model_name}-{exp_args.subset}", date)
    create_if_not_exists(training_args.output_dir)
    
    training_args.run_name = f"dpo-zephyr-{exp_args.subset}-{date}"

    full_config = {"args": vars(copy(args)), "training_args": vars(copy(training_args)), "model_config": vars(copy(model_config)), "exp_args": vars(copy(exp_args))}
    save_as_json(training_args.output_dir, "full_config.json", full_config)
        
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = torch.bfloat16
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
        ref_model = ref_model.eval()
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.padding_side = 'right'
        tokenizer.pad_token=tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = mix_datasets([float(p) for p in exp_args.dataset_proportions.split(",")], exp_args.subset)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))


    def process(row):
        if random.random() <= exp_args.use_chat_template_p:
            row["prompt"] = tokenizer.apply_chat_template([{"role":"user","content":row["prompt"]}], tokenize=False)
            row["chosen"] = tokenizer.apply_chat_template([{"role":"assistant","content":row["chosen"]}], tokenize=False)
            row["rejected"] = tokenizer.apply_chat_template([{"role":"assistant","content":row["rejected"]}], tokenize=False)
        return row

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        ds = ds.map(process, num_proc=training_args.dataset_num_proc)
        
    ds = ds.train_test_split(test_size=0.05)

    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split]

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    print("Waited")
    
    # benchmark model
    benchmark = Benchmark(output_dir=training_args.output_dir,
                        tasks = ["mmlu", "wmdp-bio", "wmdp-chem", "wmdp-cyber"],
                        wandb_project="scratch",
                        run_name=training_args.run_name,
                        upload_requests_to_hf=False,
                        save_requests=False,
                        config= full_config)
                        
    benchmark.run(trainer.model, tokenizer)
    
    trainer.accelerator.wait_for_everyone()
    # save model
    trainer.save_model(training_args.output_dir)
    # push to hub
    trainer.push_to_hub(f"{HF_USERNAME}/{training_args.run_name}_trainer")
        