import os
import json
import itertools
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarking.benchmark import Benchmark
from src.util.globals import OUTPUT_DIR
from src.util.helpers import create_if_not_exists, save_as_json
from src.logit_lens.logit_lens_utils import Llama2Wrapper


def prepare_requests(args) -> dict:
    benchmark = Benchmark(os.path.join(OUTPUT_DIR, "logit_lens"),
                        wmdp_adv_prefix="Answer the following question with A, B, C, or D.\n\n",
                        tasks=[args.task],
                        upload_requests_to_hf=False,
                        save_requests=False,)
    context = benchmark.get_context(args.task)
    dataset = load_dataset("cais/wmdp", args.task, split="test")
    requests = benchmark.generate_wmdp_requests(dataset, context)
    return requests

def collect_responses(model, tokenizer, requests):
    n_layers = len(model.model.model.layers)
    act_types = ["block", "mlp", "inter_res", "attention"]
    answers = list("ABCD")
    tokenized_ans = tokenizer(answers, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
    results = {act_t : {layer: {pos : [] for pos in range(len(answers))} for layer in range(n_layers)} for act_t in act_types}
    combinations = list(itertools.product(act_types, range(n_layers), range(len(answers))))
    
    for req in tqdm(requests.values()):
        out = model.decode_nlayers(req["question"], n=n_layers, search_tokens=tokenized_ans, topk=None, print_mode=False)
        
        for act_t, layer, position in combinations:
            pred = out[act_t][layer][position][0]
            results[act_t][layer][position].append(pred == answers[req["answer_idx"]])
    
    # we use this format bc it's easier to use with pandas
    list_of_dicts = []
    for act_t, layer, position in combinations:
        acc = np.mean(results[act_t][layer][position])
        list_of_dicts.append({"act_type": act_t, "layer": layer, "position": position, "accuracy": acc})
    
    return list_of_dicts

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--task", choices=["wmdp-bio", "wmdp-cyber"])
    parser.add_argument("--apply_chat_template", action="store_true")
    args = parser.parse_args()
    
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    
    # make output directory
    chat_str = "_chat" if args.apply_chat_template else ""
    result_dir = os.path.join(OUTPUT_DIR, "logit_lens", f"results{chat_str}")
    create_if_not_exists(result_dir)
    
    # prepare dataset
    requests = prepare_requests(args)
    
    # load model and tokenizer 
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 device_map = "auto",
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16,)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model = Llama2Wrapper(hf_model, tokenizer, use_chat_template=args.apply_chat_template)
    
    # collect responses
    results = collect_responses(model, tokenizer, requests)
    for r in results:
        r.update({"model": args.model_name_or_path.split("/")[-1], "task": args.task})
        
    # save results
    with open(os.path.join(result_dir, f"results.jsonl"), "a") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")
    
    