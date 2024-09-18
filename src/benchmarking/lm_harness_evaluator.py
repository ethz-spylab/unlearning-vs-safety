from typing import Optional, Union
from statistics import mean

import transformers
import lm_eval
from lm_eval.models.huggingface import HFLM

class HarnessEvaluator:
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: Union[
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ],
        tasks: Union[list[str], str],
        num_fewshot: Optional[int] = 0,
        batch_size: Optional[int] = 6,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = "auto",
        limit: Optional[Union[int, float]] = None,
        cache_requests: bool = True,
        apply_chat_template: bool = False,
        random_seed: int = 42,
        numpy_random_seed: int = 42,
        torch_random_seed: int = 42,
        fewshot_random_seed: int = 42,
        ) -> None:
        
        self.lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            trust_remote_code=True,
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
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.limit = limit
        self.device = device
        self.cache_requests = cache_requests
        self.random_seed = random_seed
        self.numpy_random_seed = numpy_random_seed
        self.torch_random_seed = torch_random_seed
        self.fewshot_random_seed = fewshot_random_seed
        self.apply_chat_template = apply_chat_template
    
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
            apply_chat_template = self.apply_chat_template,
            random_seed=self.random_seed,
            numpy_random_seed = self.numpy_random_seed,
            torch_random_seed = self.torch_random_seed,
            fewshot_random_seed = self.fewshot_random_seed,
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
        
        return clean_results
        