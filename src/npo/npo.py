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

import logging
import multiprocessing
import os
import datetime
import random
from copy import copy
from contextlib import nullcontext
from typing import Literal
from accelerate import Accelerator
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from trl.env_utils import strtobool

from src.npo.trainer import NPOTrainer, DataCollatorForLanguageModellingWithRetain
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
from datasets import load_dataset, concatenate_datasets
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

@dataclass
class ExperimentConfig:
    subset: Literal["bio", "cyber"] = field(default="bio", metadata={"help": "bio or cyber subset for refusal training"})
    use_chat_template_p: float = field(default=0.5, metadata={"help": "probability of using applying the chat template to each sample"})
    retain_mult: float = field(default=5, metadata={"help": "Multiplier applied to retain loss"})


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig, ExperimentConfig))
    args, training_args, model_config, exp_args= parser.parse_args_and_config()
       
    
    # make sure the output directory exists
    date = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    formatted_model_name = model_config.model_name_or_path.replace("/", "-")
    training_args.output_dir = os.path.join(OUTPUT_DIR, f"npo-{formatted_model_name}-{exp_args.subset}", date)
    create_if_not_exists(training_args.output_dir)
    
    training_args.run_name = f"npo-zephyr-{exp_args.subset}-{date}"

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
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the NPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_dataset(f"J4Q8/{exp_args.subset}_forget_dpo", split="train")
    
    # prepare retain dataset
    mc_retain = load_dataset(f"J4Q8/mc_{exp_args.subset}_retain_dpo", split="train")
    oass = get_open_assistant_dataset()
    retain_data = concatenate_datasets([mc_retain, oass]).shuffle(42).take(len(ds))
    
    # add retain dataset
    ds = ds.add_column("prompt_retain", retain_data["prompt"])
    ds = ds.add_column("chosen_retain", retain_data["chosen"])
    
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))


    def process(row):
        if random.random() <= exp_args.use_chat_template_p:
            ids_forget = tokenizer.apply_chat_template([{"role":"user","content":row["prompt"]},
                                                        {"role":"assistant", "content":row["rejected"]}], tokenize=True)
            ids_retain = tokenizer.apply_chat_template([{"role":"user","content":row["prompt_retain"]},
                                                        {"role":"assistant", "content":row["chosen_retain"]}], tokenize=True)
        else:
            ids_forget =  tokenizer(row["prompt"] + " " + row["rejected"]).input_ids
            ids_retain =  tokenizer(row["prompt_retain"] + " " + row["chosen_retain"]).input_ids
        return {"forget": {"input_ids": ids_forget}, "retain": {"input_ids": ids_retain}}

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        ds = ds.map(process, num_proc=training_args.dataset_num_proc, remove_columns=ds.column_names)
        
    ds = ds.train_test_split(test_size=0.05)

    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split]
    
    data_collator = DataCollatorForLanguageModellingWithRetain(tokenizer, mlm=False)

    ################
    # Training
    ################
    with init_context:
        trainer = NPOTrainer(
            ref_model,
            retain_mult=exp_args.retain_mult,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    print("Waited")
    
    # benchmark model # parellelization is not implemented -> inefficient
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
        