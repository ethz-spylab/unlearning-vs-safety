import dataclasses
import datetime
import json
import os
import pickle
import sys
import warnings

import datasets
import pandas as pd
import peft
import torch
import transformers
import trl

import cutil
import flrt.modal_defs as modal_defs
import flrt.templates as templates
import flrt.util as util


@dataclasses.dataclass(slots=True)
class FinetuneConfig:
    model_config: util.ModelConfig
    run_name: str
    project_name: str = "flrt-finetune"
    last_layer: int = None
    wandb_log: bool = True
    dataset_frac: float = 1.0


@modal_defs.stub.function(**modal_defs.default_params)
def finetune(c: FinetuneConfig):
    util.hf_login()
    os.environ["WANDB_PROJECT"] = c.project_name
    mc = c.model_config

    ########################################
    # DATASET
    ########################################
    phase = "dev"
    behaviors = json.load(open(f"data/{phase}_behaviors.json", "r"))
    sample_instances = pickle.load(open(f"data/{phase}_sample_instances.pkl", "rb"))

    rows = []
    for b in behaviors:
        for si in sample_instances[b]:
            rows.append({"prompt": b, "completion": si})
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(rows))
    train_test_split = dataset.train_test_split(test_size=0.025, shuffle=True, seed=0)
    train_dataset = train_test_split["train"]
    if c.dataset_frac < 1.0:
        train_dataset = train_dataset.select(
            range(int(len(train_dataset) * c.dataset_frac))
        )
    eval_dataset = train_test_split["test"]

    tokenizer = util.load_tokenizer(mc.model_name)

    def format_prompt(example, include_completion=True):
        output_texts = []
        for i in range(len(example["prompt"])):
            output = templates.build_prompt(
                tokenizer,
                templates.build_one_turn_chat(mc.system_prompt, example["prompt"][i]),
            )
            if include_completion:
                output += example["completion"][i]
            output_texts.append(output)
        return output_texts

    collator = trl.DataCollatorForCompletionOnlyLM(
        mc.response_template, tokenizer=tokenizer
    )

    ########################################
    # MODEL
    ########################################
    # NOTE: training in float16 often causes nans.
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        mc.model_name, **model_kwargs
    )

    peft_config = peft.LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        layers_to_transform=(
            None if c.last_layer is None else list(range(c.last_layer + 1))
        ),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ],
    )

    ########################################
    # TRAIN
    ########################################
    trainer = trl.SFTTrainer(
        model,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=format_prompt,
        data_collator=collator,
        args=trl.SFTConfig(
            max_seq_length=2048,
            eval_strategy="steps",
            eval_steps=100,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            logging_dir="./logs",
            logging_steps=1,
            output_dir="./results",
            report_to="wandb" if c.wandb_log else "none",
            run_name=c.run_name,
        ),
    )
    trainer.train()
    trainer.save_model(f"./output/{c.project_name}/{c.run_name}")

    ########################################
    # TEST
    ########################################

    behaviors = json.load(open("data/test_behaviors.json", "r"))
    rows = []
    for b in behaviors:
        rows.append({"prompt": b})
    test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(rows))
    test_prompts = [
        p + mc.first_token
        for p in format_prompt(test_dataset, include_completion=False)
    ]
    for i in range(50):
        print("PROMPT:", repr(test_dataset[i]["prompt"]))
        inputs = tokenizer(
            test_prompts[i : i + 1], add_special_tokens=False, return_tensors="pt"
        )
        all_ids = trainer.model.generate(
            **{k: v.to("cuda") for k, v in inputs.items()},
            max_new_tokens=64,
            do_sample=False,
        )
        print(
            "RESPONSE:",
            repr(tokenizer.decode(all_ids[0, inputs["input_ids"].shape[1] :])),
            "\n",
        )


def modal_finetune(cfgs):
    use_modal = True
    try:
        is_debug_run = sys.gettrace() is not None or "profile" in __builtins__
    except TypeError:
        is_debug_run = False
    if is_debug_run:
        warnings.warn("Debug run! wandb_log = False. Running locally.")
        use_modal = False
        for c in cfgs:
            c.wandb_log = False

    if use_modal:
        with modal_defs.stub.run():
            list(finetune.map(cfgs))
    else:
        for c in cfgs:
            finetune.local(c)


# earlysoft-Llama-2-7b-chat-hf-240621_052448 - 100% of data, up to layer 0
# earlysoft-Llama-2-7b-chat-hf-240621_051852 - 25% of data, up to layer 1
# earlysoft-Llama-2-7b-chat-hf-240621_051445 - 25% of data, up to layer 2
# earlysoft-Llama-2-7b-chat-hf-240621_051041 - 25% of data, up to layer 4
# earlysoft-Llama-2-7b-chat-hf-240621_042722 - 25% of data, up to layer 8
# earlysoft-Llama-3-8B-Instruct-RR-240711_152536 - 100% of data, up to layer 4, based on the grayswan circuit breaker model
# earlysoft-Llama-3-8B-Instruct-RR-240711_153553 - 100% of data, up to layer 8, based on the grayswan circuit breaker model
def main_early():
    cutil.chdir_git_root("flrt")
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    mcd = dataclasses.asdict(util.model_configs["llama-3"])
    mcd["model_name"] = "GraySwanAI/Llama-3-8B-Instruct-RR"
    mc = util.ModelConfig(**mcd)
    model_subname = mc.model_name.split("/")[-1]
    run_name = f"earlysoft-{model_subname}-{now}"
    cfgs = [
        FinetuneConfig(
            model_config=mc,
            run_name=run_name,
            last_layer=8,
            dataset_frac=1.0,
        )
    ]
    modal_finetune(cfgs)


def main_full():
    cutil.chdir_git_root("flrt")
    names = ["llama-2", "llama-3", "phi-3", "gemma-7b", "gemma-2b", "vicuna"]
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    cfgs = []
    for name in names:
        mc = util.model_configs[name]
        model_subname = mc.model_name.split("/")[-1]
        run_name = f"{model_subname}-{now}"
        cfgs.append(FinetuneConfig(model_config=mc, run_name=run_name))
    modal_finetune(cfgs)


if __name__ == "__main__":
    main_early()
