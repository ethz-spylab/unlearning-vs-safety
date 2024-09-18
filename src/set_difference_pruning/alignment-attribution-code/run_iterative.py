import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import gc         # garbage collect library
import numpy as np
import torch
import json
import shutil
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import (
    prune_wanda,
    prune_random,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    check_sparsity,
    check_sparsity_layerwise,
    find_layers,
    prune_wanda_decouple_activations,
    get_mask,
    prune_wandg_set_difference,
    prune_attention_head,
)
from lib.model_wrapper import prune_wanda_v2, prune_wandg, prune_layer_by_layer
from lib.model_wrapper_low import make_low_rank
from lib.constants import MODEL_CACHE_DIR, RESULTS_DIR
from lib.experiment import Experiment
from lib.finetune_module import FinetuningModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_sample_limit", type=int)
    parser.add_argument("--use_finetuning", action="store_true")
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
                        default="mmlu,wmdp,utility")
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
    parser.add_argument(
        "--iter_num",
        type=int,
        default=4,
        help = "number of iterations of pruning"
    )

    # low rank
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--niter", type=int, default=20)

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
    
    # Dictionary to store configuration and results
    info_dict = {}
    info_dict.update(vars(args))
    
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
    
    FULL_DICT = {
        "llama2-13b-chat-hf": {
            "full_name": "meta-llama/Llama-2-13b-chat-hf",
            "tokenizer": "meta-llama/Llama-2-13b-chat-hf"
        },
        "llama2-7b-chat-hf": {
            "full_name": "meta-llama/Llama-2-7b-chat-hf",
            "tokenizer": "meta-llama/Llama-2-7b-chat-hf"
        },
        "zephyr-7b-beta": {
            "full_name": "HuggingFaceH4/zephyr-7b-beta",
            "tokenizer": "HuggingFaceH4/zephyr-7b-beta"
        },
    }
    
    def get_model_tokenizer(model_str:str = "", max_length = 1024) -> tuple:
        model = AutoModelForCausalLM.from_pretrained(
                FULL_DICT[model_str]["full_name"],
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_CACHE_DIR,
                low_cpu_mem_usage=True,
                device_map="cuda:0",
            )
        model.eval()
        model.config.max_length = max_length # by default it is 20 which is weirds
        model.seqlen = max_length
        tokenizer = AutoTokenizer.from_pretrained(
            FULL_DICT[model_str]["tokenizer"], use_fast=False,
            )
        return model, tokenizer
    
    N_iter = args.iter_num
    experiment_directory = args.save
    if args.layer_by_layer:
        experiment_directory += "_layer_by_layer"
    prune_data = args.prune_data
    util_data = args.util_data
    last_model_path = ""
    for i in range(N_iter):
        info_dict[f"iter_{i}"] = {}
        iter_dir = os.path.join(experiment_directory, f"iter_{i}")
        
        if i == 0:
            model, tokenizer = get_model_tokenizer(args.model)
        else:
            # remove previous model for memory efficiency
            path_to_previous_model = os.path.join(experiment_directory, f"iter_{i-1}", "pruned_model", args.model)
            shutil.rmtree(path_to_previous_model)
            
            # apply finetuning if specified
            if args.use_finetuning:
                gpu_stats = torch.cuda.get_device_properties(0)
                start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
                max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
                print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
                print(f"{start_gpu_memory} GB of memory reserved.")
                print(f"=== Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB ===")
                print(model.training)
                model = FinetuningModule().finetune(model, tokenizer, i)
                model.eval()
                print(model.training)
                print("========= AFTER FINETUNING =========")
                used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
                used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
                used_percentage = round(used_memory         /max_memory*100, 3)
                lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
                print(f"Peak reserved memory = {used_memory} GB.")
                print(f"=== Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB ===")
                print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
                print(f"Peak reserved memory % of max memory = {used_percentage} %.")
                print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
                    
        if args.layer_by_layer:
            print("Using layer by layer")
            args.save = iter_dir
            prune_layer_by_layer(
                args, 
                model,
                tokenizer,
                dataset_seed=i
            )
        else:
            compute_scores(model, tokenizer, args, prune_data, iter_dir, i, args.prune_method)
            compute_scores(model, tokenizer, args, util_data, iter_dir, i, args.prune_method)
            
            args.save = iter_dir
            prune_wandg_set_difference(
                args,
                model,
                tokenizer,
                prune_data=args.prune_data,
                p=args.p,
                q=args.q,
                dynamic_sparsity=args.dynamic_sparsity_allocation,
                layer_subset=args.layer_subset,
                directory = iter_dir,
                scoring_method = args.prune_method,
            )
            
            # delete scores for memory efficiency
            prev_scores_dir = os.path.join(iter_dir, args.model, "unstructured", args.prune_method)
            shutil.rmtree(prev_scores_dir)
              
        last_model_path = os.path.join(iter_dir, "pruned_model", args.model)
        model.save_pretrained(last_model_path)
        tokenizer.save_pretrained(last_model_path)
        print(f"Model saved to: {last_model_path}")

        print(torch.cuda.memory_allocated(0))
        ################################################################
        print("*" * 30)
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.6f}")
        info_dict[f"iter_{i}"]["sparsity_ratio"] = sparsity_ratio
        ################################################################
        if len(args.tasks) != 0:
            experiment = Experiment(model, tokenizer, tasks=args.tasks, limit=args.task_sample_limit)
            results = experiment.run()
            info_dict[f"iter_{i}"].update(results)
        ######################################################
        # save the results after each iter to make sure its going well
        result_path = os.path.join(iter_dir, f"result_{args.prune_method}_{args.dynamic_sparsity_allocation}_disentangled_data.jsonl")
        mode = "a" if os.path.isfile(result_path) else "w"
        with open(result_path, mode) as f:
            json.dump(info_dict, f)
            f.write('\n')
    
        
if __name__ == "__main__":
    main()