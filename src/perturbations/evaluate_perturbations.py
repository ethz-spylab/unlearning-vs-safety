import json
import os
import re
from argparse import ArgumentParser

import torch
import transformers

from src.benchmarking.benchmark import Benchmark
from src.util.globals import OUTPUT_DIR

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--perturbations_path", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    
    # get request files
    files = os.listdir(args.perturbations_path)
    json_files = [file for file in files if file.endswith(".json")]
    
    # load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                                low_cpu_mem_usage=True,
                                                                torch_dtype=torch.bfloat16,
                                                                device_map='auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,)
    
    for file in json_files:            
        partial, total = re.search(r"(\d.\d)_(\d.\d).json", file).groups((1,2))
        
        # run benchmark
        benchmark = Benchmark(output_dir=os.path.join(OUTPUT_DIR, "perturbations", "informed", f"{args.model_name_or_path.split('/')[-1]}", "wmdp-bio"),
                            tasks=["wmdp-bio"],
                            wandb_project="perturbations-informed",
                            run_name=f"{partial}_{total}",
                            upload_requests_to_hf=False,
                            save_requests=False,
                            config={"model": args.model_name_or_path.split("/")[-1],
                                    "partial": partial,
                                    "total": total},
                            request_file={"wmdp-bio": os.path.join(args.perturbations_path, file)},)
        benchmark.run(model, tokenizer)