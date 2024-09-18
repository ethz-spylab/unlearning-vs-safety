from typing import Optional, Union
from statistics import mean

import torch
import transformers
import lm_eval
from lm_eval.models.huggingface import HFLM

from lib.eval import eval_ppl_wikitext
from lib.data import get_loaders

class Experiment:
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: Union[
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ],
        tasks: Union[list[str], str],
        evaluate_perplexity: bool = False,
        num_fewshot: Optional[int] = 0,
        batch_size: Optional[int] = 6,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = "cuda:0",
        limit: Optional[Union[int, float]] = None,
        cache_requests: bool = True,
        ) -> None:
        
        self.lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
        # if tasks are passed as string with values delimited by comas turn it into a list
        if isinstance(tasks, str):
            tasks = tasks.split(",")
        
        # process utility tasks
        self.utility_tasks = None
        if "utility" in tasks:
            self.utility_tasks = [
                "boolq",
                "rte",
                "hellaswag",
                "winogrande",
                "arc_challenge",
                "openbookqa"
            ]
            tasks.remove("utility")
            tasks.extend(self.utility_tasks)
        
        self.tasks = tasks
        self.evaluate_perplexity = evaluate_perplexity
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.limit = limit
        self.device = device
        self.cache_requests = cache_requests
    
    def run(self) -> dict:
        clean_results = {}
        
        results = lm_eval.simple_evaluate(
            model = self.lm,
            batch_size=self.batch_size,
            tasks = self.tasks,
            num_fewshot=self.num_fewshot,
            device=self.device,
            limit=self.limit,
            max_batch_size = self.max_batch_size,
            cache_requests = self.cache_requests,
        )["results"]
        
        for task in self.tasks:
            # in case of wmdp we need to add individual scores
            if task == "wmdp":
                for t in ["wmdp_bio", "wmdp_chem", "wmdp_cyber"]:
                    clean_results[t] = results[t]["acc,none"]
                continue
            # do not report utility tasks individually but only as average
            if self.utility_tasks is not None and task in self.utility_tasks:
                continue
            # just add all the other results
            clean_results[task] = results[task]["acc,none"]
        
        # compute average of utility tasks
        if self.utility_tasks is not None:
            avg = mean([results[task]["acc,none"] for task in self.utility_tasks])
            clean_results["utility"] = avg
        
        # evaluate ppl
        if self.evaluate_perplexity:
            print(f"Evaluating perplexity on wikitext")
            model = self.lm._model
            tokenizer = self.lm.tokenizer
            # Get the test loader
            _, testloader = get_loaders(
                "wikitext", seed=0, seqlen=model.seqlen, tokenizer=tokenizer
            )
            
            # Evaluate ppl in no grad context to avoid updating the model
            with torch.no_grad():
                ppl_test = eval_ppl_wikitext(model, testloader, 1, self.device)
            
            clean_results["ppl"] = ppl_test
        
        return clean_results
        