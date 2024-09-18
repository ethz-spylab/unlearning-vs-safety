import argparse
import os
import gc
from turtle import up
import numpy as np
from sympy import im
import torch
import json
import shutil
from itertools import product
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM


from .lib.prune import check_sparsity, prune_wandg_set_difference
from .lib.model_wrapper import prune_wanda_v2, prune_wandg

from src.benchmarking.benchmark import Benchmark
from src.util.globals import OUTPUT_DIR

### example
"""python run_gridsearch.py --model zephyr-7b-beta --save "/home/jlucki/project_data/wmdp_localization/gridsearch" --prune_method wandg"""

FULL_DICT = {
        "llama2-13b-chat-hf": {
            "full_name": "meta-llama/Llama-2-13b-chat-hf",
            "tokenizer": "meta-llama/Llama-2-13b-chat-hf"
        },
        "zephyr-7b-beta": {
            "full_name": "HuggingFaceH4/zephyr-7b-beta",
            "tokenizer": "HuggingFaceH4/zephyr-7b-beta"
        },
        "cais/Zephyr_RMU":
        {
            "full_name": "cais/Zephyr_RMU",
            "tokenizer": "cais/Zephyr_RMU"
        },
        "J4Q8/zephyr-dpo-bio":
        {
            "full_name": "J4Q8/zephyr-dpo-bio",
            "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
        },
        "J4Q8/zephyr-npo-bio":
        {
            "full_name": "J4Q8/zephyr-npo-bio",
            "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
        },
        "J4Q8/zephyr-dpo-cyber":
        {
            "full_name": "J4Q8/zephyr-dpo-cyber",
            "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
        },
        "J4Q8/zephyr-npo-cyber":
        {
            "full_name": "J4Q8/zephyr-npo-cyber",
            "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
        }
    }

def create_if_not_exists(path:str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
            
def compute_scores(model, tokenizer, args, dataset:str, dir: str, iteration: int, scoring_method: Literal["wanda", "wandg"]) -> None:
    methods = {
        "wanda": prune_wanda_v2,
        "wandg": prune_wandg,
    }
    args.dump_wanda_score = True
    args.save = os.path.join(dir, args.model, "unstructured", scoring_method, dataset)
    methods[scoring_method](
        args = args,
        model = model,
        tokenizer = tokenizer,
        prune_data=dataset,
        dataset_seed=iteration, # thanks to this we will select different parts of the dataset at every iteration
    )
    args.dump_wanda_score = False
    
def get_model_tokenizer(model_str:str = "", max_length = 1024) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
            FULL_DICT[model_str]["full_name"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    model.eval()
    model.config.max_length = max_length # by default it is 20 which is weirds
    model.seqlen = max_length
    tokenizer = AutoTokenizer.from_pretrained(
        FULL_DICT[model_str]["tokenizer"], use_fast=False,
        )
    return model, tokenizer

NSAMPLES = 512
LAYER_SUBSETS = [None] #[None, "mlp_only", "omit_keys_queries"]
DYNAMIC_SPARSITY = [False]
P_LIST = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, .125, 0.15, 0.2]
Q_LIST = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, .125, 0.15, 0.2]
# PRUNE_DATASETS = ["wmdp_bio-forget-corpus_abstract", "wmdp_bio-forget-corpus"]
# UTILITY_DATASETS = ["wikitext_mmlu", "wmdp_bio-retain-corpus_mmlu"] #["wikitext", "wikitext_mmlu", "wmdp_bio-retain-corpus", "wmdp_bio-retain-corpus_mmlu"]

def get_gridsearch_configs() -> list[dict]:
    configs = []
    for p, q in product(P_LIST, Q_LIST):
        # skip cases where pruning param is larger than util_param as this usually leads to worse results
        if p < q:
            continue
        configs.append({
            "layer_subset": None,
            "dynamic_sparsity_allocation": False,
            "p": p,
            "q": q,
            "nsamples": NSAMPLES,
        })
    print(f"Created {len(configs)} configurations")
    return configs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", "-i", type=int, help="which configuration to use, if None all will be used (currently causes OOMs)")
    parser.add_argument("--start_id", type=int, default=0, help="from which configuration to start gridsearch (OOMs still happening)")
    parser.add_argument("--layer_by_layer", action="store_true")
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--model_base", type=str, default="llama2-7b-hf")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type",
        type=str,
        choices=["unstructured", "4:8", "2:4"],
        default="unstructured",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "wanda",
            "wandg",
        ],
        help="Set difference will be applied applied anyway, this only allows to decide which kind of score to compute"
    )
    parser.add_argument(
        "--dynamic_sparsity_allocation",
        action="store_true",
        help="whether to use sparsity rates optimized per layer"
    )
    parser.add_argument(
        "--layer_subset",
        choices=["mlp_only", "omit_keys_queries"],
        help="select a subset of layers that will be pruned, all layers will be pruned if not specified"
    )
    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "wikitext",
            "wikitext_mmlu",
            "alpaca",
            "alpaca_cleaned",
            "alpaca_cleaned_no_safety",
            "align",
            "align_short",
            "misalign",
            "align_misalign",
            "misalign_align",
            "align_short_misalign",
            "wmdp_bio-retain-corpus",
            "wmdp_bio-retain-corpus_mmlu",
            "wmdp_bio-forget-corpus",
            "wmdp_bio-forget-corpus_abstract",
            "wmdp_cyber-forget-corpus",
            "wmdp_cyber-retain-corpus",
            "wmdp_cyber-retain-corpus_mmlu",
            "mmlu",
            "mmlu_abcd",
            "math__add_sub",
            "math__mul_div",
            "generations-Zephyr_RMU-wmdp-bio-prune",
            "generations-zephyr-npo-bio-wmdp-bio-prune",
            "generations-zephyr-dpo-bio-wmdp-bio-prune",
            "generations-Zephyr_RMU-wmdp-cyber-prune",
            "generations-zephyr-npo-cyber-wmdp-cyber-prune",
            "generations-zephyr-dpo-cyber-wmdp-cyber-prune",
            "none",
        ],
        default="alpaca_cleaned_no_safety",
    )
    parser.add_argument("--util_data", type=str, choices=[
        "wikitext",
        "wikitext_mmlu",
        "mmlu",
        "mmlu_abcd",
        "alpaca",
        "alpaca_cleaned",
        "alpaca_cleaned_no_safety",
        "wmdp_bio-retain-corpus",
        "wmdp_bio-retain-corpus_mmlu",
        "wmdp_bio-forget-corpus",
        "wmdp_bio-forget-corpus_abstract",
        "wmdp_cyber-forget-corpus",
        "wmdp_cyber-retain-corpus",
        "wmdp_cyber-retain-corpus_mmlu",
        "math__add_sub",
        "math__mul_div",
        "generations-Zephyr_RMU-wmdp-bio-util",
        "generations-zephyr-npo-bio-wmdp-bio-util",
        "generations-zephyr-dpo-bio-wmdp-bio-util",
        "generations-Zephyr_RMU-wmdp-cyber-util",
        "generations-zephyr-npo-cyber-wmdp-cyber-util",
        "generations-zephyr-dpo-cyber-wmdp-cyber-util",
        ],
        default="wikitext",
        help="Only for wandg_set_difference"
    )
    parser.add_argument("--use_diff", action="store_true")
    parser.add_argument("--neg_prune", action="store_true")
    parser.add_argument("--recover_from_base", action="store_true")
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top p scored elements in the first set (alpaca_no_safety)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top q scored elements in the second set (align))",
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        default=10,
        help="Use combined with attention_head, the top k heads to prune",
    )

    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the pruned model."
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=None,
        help="Path to save the pruned model weight mask.",
    )
    parser.add_argument(
        "--dump_wanda_score", action="store_true", help="Whether to dump wanda scores."
    )

    parser.add_argument("--tasks", 
                        type=str, 
                        help="coma separated list of tasks to be evaluated", 
                        default="mmlu,wmdp-bio,wmdp-cyber,wmdp-chem")
    parser.add_argument(
        "--prune_part",
        action="store_true",
        help="whether to only prune the layer with lower jaccard index",
    )
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_utility",
        action="store_true",
        help="whether to decouple the align and utility when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_misalign",
        action="store_true",
        help="whether to decouple the align and misalign when computing the wanda score",
    )
    args = parser.parse_args()

    print("Disentangle:", args.disentangle)

    if args.dump_wanda_score:
        assert args.prune_method in [
            "wanda",
            "wanda_v2",
            "wandg",
        ], "dump_wanda_score only works with wanda wanda_v2 wandg"

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    
    configs = get_gridsearch_configs()
    experiment_directory = os.path.join(OUTPUT_DIR, "set_difference_pruning")
    model, tokenizer = get_model_tokenizer(args.model)
    
    # if config id is set then use only a particular config
    compute_scores_flag = False
    if args.config_id is not None:
        configs = configs[args.config_id:args.config_id+1]
        print(f"RUNNING only confing no.{args.config_id}")
        if args.config_id == 0:
            compute_scores_flag = True
    else:
        print(f"RUNNING all confings starting with no.{args.start_id}")
        configs = configs[args.start_id:]
        if args.start_id == 0:
            compute_scores_flag = True
    
    for id, config in enumerate(configs):
        # set new parameters using config
        vars(args).update(config)
        print(args)
        # Dictionary to store configuration and results
        info_dict = {}
        info_dict.update(vars(args))
        
        if compute_scores_flag and id == 0:   
            # compute all necessary scores
            for dataset in [args.util_data, args.prune_data]:
                compute_scores(model, tokenizer, args, dataset, experiment_directory, args.seed, args.prune_method)
                
        if id > 0:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            # reload model for every config
            model, tokenizer = get_model_tokenizer(args.model)
        
        prune_wandg_set_difference(
                args,
                model,
                tokenizer,
                prune_data=args.prune_data,
                p=args.p,
                q=args.q,
                dynamic_sparsity=args.dynamic_sparsity_allocation,
                layer_subset=args.layer_subset,
                directory = experiment_directory,
                scoring_method = args.prune_method,
            )
        
        print(torch.cuda.memory_allocated(0))
        ################################################################
        print("*" * 30)
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.6f}")
        info_dict["sparsity_ratio"] = sparsity_ratio
        ################################################################
        args.tasks = args.tasks.split(",")
        if len(args.tasks) != 0:
            bench = Benchmark(
                output_dir=os.path.join(experiment_directory, "results"),
                tasks=args.tasks,
                wandb_project="set_difference_pruning",
                run_name=f"{args.model}_p{args.p}_q{args.q}",
                save_requests=False,
                upload_requests_to_hf=False,
                config=info_dict,
            )
            bench.run(model, tokenizer)
            bench.finish()
            del bench
    
        
if __name__ == "__main__":
    main()