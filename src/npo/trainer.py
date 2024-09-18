from types import MethodType
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, List, Union, Any, Dict, Mapping
from torch import nn
import torch
from transformers import Trainer, DataCollatorForLanguageModeling
import torch.nn.functional as F
import deepspeed

"""Adapted from: https://github.com/jinzhuoran/RWKU/blob/main/LLaMA-Factory/src/llmtuner/train/npo/trainer.py"""


class DataCollatorForLanguageModellingWithRetain(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        forget = super().torch_call([example["forget"] for example in examples])
        retain = super().torch_call([example["retain"] for example in examples])
        return {"forget":forget, "retain":retain}


class NPOTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(self, ref_model, retain_mult, **kwargs) -> None:
        super().__init__(**kwargs)
        self.retain_mult=retain_mult
        if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(ref_model)
        else:
            self.ref_model = ref_model
    
    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_batch_loss(self, output, labels):
        shifted_labels = labels[..., 1:].contiguous()
        output = output[..., :-1, :].contiguous()

        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # get the sum loss for each sequence in a batch
        loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # npo loss
        inputs_f = inputs["forget"]
        _, outputs = super().compute_loss(model, inputs_f, True)
        forget_loss_current = self.get_batch_loss(outputs.logits, inputs_f['labels'])
        with torch.no_grad():
            forget_outputs_oracle = self.ref_model(inputs_f['input_ids'], labels=inputs_f['labels'], attention_mask=inputs_f['attention_mask'])
            forget_logits_oracle = forget_outputs_oracle.logits
            forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, inputs_f['labels'])
        neg_log_ratios = forget_loss_current - forget_loss_oracle
        forget_loss = -F.logsigmoid(self.args.beta * neg_log_ratios).mean() * 2 / self.args.beta
        # retain loss
        inputs_r = inputs["retain"]
        _, outputs_r = super().compute_loss(model, inputs_r, True)
        retain_loss = self.get_batch_loss(outputs_r.logits, inputs_r['labels']).mean()
        
        loss = forget_loss + self.retain_mult*retain_loss
        
        return (loss, outputs) if return_outputs else loss