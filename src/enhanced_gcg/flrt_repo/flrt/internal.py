import contextlib
import dataclasses
import logging
import gc
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
from sympy import Nor, use
import torch
from transformers import PreTrainedModel

from .objective import Objective, TaskData
from .util import load_model

logger = logging.getLogger()


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


@dataclasses.dataclass
class InternalObjective(Objective):
    attack_layer: int = 20

    def setup(self, new_victim):
        assert self.attack_mult > 0
        out = super().setup(new_victim, skip_ft_probs=True)
        out.attack_layers = [out.attack_layer]

        for task_data in out.task_data:
            if task_data.n_match > 0:
                # self.attack_layers = list(range(10, 32))

                task_data.ft_resid, _ = out.calc_resid(
                    input_ids=torch.cat(
                        (task_data.ft_input_ids, task_data.match_ids)
                    ).unsqueeze(0),
                    attack_layers=out.attack_layers,
                    use_ft_model=True,
                )
                task_data.ft_resid = task_data.ft_resid[
                    :, :, -task_data.match_ids.shape[0] - 1 :
                ]
                task_data.ft_resid_norm = task_data.ft_resid.norm(
                    keepdim=True, dim=-1
                ).mean(keepdim=True, dim=-2)
        return out

    def calc_resid(self, *, attack_layers, use_ft_model, **model_kwargs):
        resid = dict()

        def cache_resid(module, input, output, layer):
            resid[layer] = input[0].to(torch.float32)

        model = (
            self.victim.ft_model if use_ft_model else self.victim.base_model.peft_model
        )
        hooks = []
        for L in attack_layers:
            hooks.append(
                (
                    model.model.model.layers[L - 1],
                    partial(cache_resid, layer=L),
                )
            )
        with contextlib.ExitStack() as stack:
            stack.enter_context(add_fwd_hooks(hooks))
            if not use_ft_model:
                stack.enter_context(model.disable_adapter())
            if model_kwargs.get("inputs_embeds", None) is None:
                model_out = model(**model_kwargs)
            else:
                del model_kwargs["input_ids"]
                model_out = model(**model_kwargs)
        return (
            torch.cat([resid[L][:, None] for L in attack_layers], dim=1),
            model_out,
        )

    def batch_attack(
        self,
        *,
        task_data: TaskData,
        prompt_ids,
        prompt_embeds,
        prompt_attention_mask,
        kv_cache=None,
        past_logits=None,
    ):
        n_batch = prompt_attention_mask.shape[0]

        model_kwargs = self.prep_args_for_model(
            task_data,
            prompt_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            kv_cache=kv_cache,
            past_logits=past_logits,
        )

        resid, model_out = self.calc_resid(
            attack_layers=self.attack_layers, use_ft_model=False, **model_kwargs
        )
        if kv_cache is None:
            logits = model_out.logits
        else:
            logits = torch.cat((past_logits, model_out.logits), dim=1)

        resid = resid[:, :, -task_data.match_ids.shape[0] - 1 :]

        vocab_size = logits.shape[-1]
        force_logits = logits[:, -task_data.force_match_ids.shape[0] - 1 :]
        force_loss = (
            torch.nn.functional.cross_entropy(
                force_logits[:, task_data.force_slice].reshape(-1, vocab_size),
                task_data.force_ids[None].repeat(n_batch, 1).ravel(),
                reduction="none",
            )
            .reshape((n_batch, -1))
            .mean(dim=-1)
        )
        force_loss = torch.clamp(force_loss, min=-np.log(self.p_threshold))

        error = ((resid - task_data.ft_resid) / task_data.ft_resid_norm) ** 2
        internal_loss = 100000 * (error.reshape(n_batch, -1).mean(dim=-1))
        attack_loss = force_loss + internal_loss
        best_idx = attack_loss.argmin()
        logger.info(
            f"internal loss components: {force_loss[best_idx]:.3f} {internal_loss[best_idx]:.3f}"
        )

        prompt_logits = logits[:, : -task_data.force_match_ids.shape[0]]
        return attack_loss, prompt_logits, logits

@dataclasses.dataclass
class InternalObjectiveOriginalModel(Objective):
    original_model: PreTrainedModel = None
    attack_layers: tuple[int] = (20)
    norm_p: int = 2
    use_cos_sim: bool = False
    use_resid_norm: bool = True # IMO it doesn't make sense to normalize it because we want the distance to be as small as possible (especially that RMU increases teh norm explicitly) and not only direction
    use_sequential_weights: bool = False
    use_static_representations: bool = False
    normalize_magnitude_across_layers: bool = False
    internal_loss_over_target_match_only: bool = False
    ignore_dynamic_representation_matching: bool = False
    dont_clamp_loss: bool = False

    def setup(self, new_victim):
        assert self.attack_mult > 0
        out = super().setup(new_victim, skip_ft_probs=True)
        
        if self.original_model is not None:

            for task_data in out.task_data:
                task_data.original_match_rep, _ = out.calc_resid(
                    input_ids=torch.cat(
                        (task_data.ft_input_ids, task_data.match_ids)
                    ).unsqueeze(0),
                    attack_layers=out.attack_layers,
                    use_ft_model=False,
                    original_model=self.original_model,
                )
                task_data.original_match_rep = task_data.original_match_rep[
                    :, :, -task_data.match_ids.shape[0] - 1:
                ]
                task_data.original_match_rep_norm = task_data.original_match_rep.norm(
                    p=self.norm_p, keepdim=True, dim=-1
                ).mean(keepdim=True, dim=-2)
                
                if self.use_sequential_weights:
                    
                    if self.internal_loss_over_target_match_only:
                        n_tokens = task_data.force_match_ids.shape[0] + 1
                    else:
                        n_tokens = task_data.match_ids.shape[0] + task_data.ft_input_ids.shape[0]
                    
                    # add more weight to the first tokens so that we first try to minimize the first tokens
                    task_data.weights = torch.linspace(2,1, n_tokens, device=task_data.original_match_rep.device)
            
            # # remove model from memory for efficiency
            # del self.original_model 
            # gc.collect()
            # torch.cuda.empty_cache()
        
        else:
            # this shouldn't really be used it is just here to check if the method works
            for task_data in out.task_data:
                task_data.ft_resid = 0
                task_data.ft_resid_norm = 1
        
        return out

    def calc_resid(self, *, attack_layers, use_ft_model, original_model=None, **model_kwargs):
        resid = dict()

        def cache_resid(module, input, output, layer):
            resid[layer] = input[0].to(torch.float32)

        model = (
            self.victim.base_model if original_model is None else original_model
        )
        hooks = []
        for L in attack_layers:
            hooks.append(
                (
                    model.model.layers[L - 1],
                    partial(cache_resid, layer=L),
                )
            )
        with contextlib.ExitStack() as stack:
            stack.enter_context(add_fwd_hooks(hooks))
            if model_kwargs.get("inputs_embeds", None) is None:
                model_out = model(**model_kwargs)
            else:
                del model_kwargs["input_ids"]
                model_out = model(**model_kwargs)


        if self.normalize_magnitude_across_layers:
            norms_list = []
            for L in attack_layers:
                norms_list.append(resid[L].norm(p=self.norm_p, keepdim=True, dim=-1))
            norms = torch.stack(norms_list, dim=0)
            maximums = norms.max(dim=0).values
            multiplier = maximums/norms
            
            for i, l in enumerate(attack_layers):
                resid[l] = resid[l]*multiplier[i]
        
        return (
            torch.cat([resid[L][:, None] for L in attack_layers], dim=1),
            model_out,
        )
        
    def batch_attack(
        self,
        *,
        task_data: TaskData,
        prompt_ids,
        prompt_embeds,
        prompt_attention_mask,
        kv_cache=None,
        past_logits=None,
    ):
        n_batch = prompt_attention_mask.shape[0]

        model_kwargs = self.prep_args_for_model(
            task_data,
            prompt_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            kv_cache=kv_cache,
            past_logits=past_logits,
        )
        
        # calculate representations on the unlearned model
        resid, model_out = self.calc_resid(
            attack_layers=self.attack_layers, use_ft_model=False, **model_kwargs
        )
        
        if kv_cache is None:
            logits = model_out.logits
        else:
            logits = torch.cat((past_logits, model_out.logits), dim=1)

        vocab_size = logits.shape[-1]
        force_logits = logits[:, -task_data.force_match_ids.shape[0] - 1 :]
        force_loss_pre_clamp = (
            torch.nn.functional.cross_entropy(
                force_logits[:, task_data.force_slice].reshape(-1, vocab_size),
                task_data.force_ids[None].repeat(n_batch, 1).ravel(),
                reduction="none",
            )
            .reshape((n_batch, -1))
            .mean(dim=-1)
        )
        
        if self.dont_clamp_loss:
            force_loss = force_loss_pre_clamp
        else:
            force_loss = torch.clamp(force_loss_pre_clamp, min=-np.log(self.p_threshold))
        
        if self.internal_loss_over_target_match_only:
            loss_start_idx = -task_data.force_match_ids.shape[0] - 1
        else:
            loss_start_idx = -(task_data.match_ids.shape[0] + task_data.ft_input_ids.shape[0])
        
        error = torch.zeros_like(force_loss)
        if not self.ignore_dynamic_representation_matching:
            
            # calculate representations on the original model
            task_data.ft_resid, _ = self.calc_resid(
                attack_layers=self.attack_layers,
                use_ft_model=False,
                original_model=self.original_model,
                **model_kwargs
            )
            task_data.ft_resid = task_data.ft_resid[
                :, :, loss_start_idx:
            ]
            task_data.ft_resid_norm = task_data.ft_resid.norm(
                p=self.norm_p, keepdim=True, dim=-1
            ).mean(keepdim=True, dim=-2)
            
            resid = resid[:, :, loss_start_idx :]
                
            error = torch.norm(resid - task_data.ft_resid, p=self.norm_p, dim=-1, keepdim=True)
            if self.use_resid_norm:
                error = error / (task_data.ft_resid_norm ** self.norm_p)
                error *= 100
            
            multiplier = 1 if self.norm_p == 2 else 0.01
            # multiplier *= len(self.attack_layers)
            
            if self.use_sequential_weights:
                # add more weight to the first tokens so that we first try to minimize the first tokens
                error = error * task_data.weights
                
            error = multiplier * error.reshape(n_batch, -1).mean(dim=-1).to(force_loss.device)
        
        original_direction_loss = torch.zeros_like(force_loss)
        if self.use_static_representations:
            match_resid = resid[:, :, -task_data.match_ids.shape[0] - 1 :]
            cos_sim = torch.nn.functional.cosine_similarity(match_resid, task_data.original_match_rep, dim=-1)
            static_penalty = 10*(1 - cos_sim)
            original_direction_loss = static_penalty.reshape(n_batch, -1).mean(dim=-1).to(force_loss.device)
            
        internal_loss = error
        attack_loss = force_loss + internal_loss + original_direction_loss
        best_idx = attack_loss.argmin()
        logger.info(
            f"internal loss components: {force_loss[best_idx]:.3f}({force_loss_pre_clamp[best_idx]:.3f}) {internal_loss[best_idx]:.3f} {original_direction_loss[best_idx]:.3f}"
        )

        prompt_logits = logits[:, : -task_data.force_match_ids.shape[0]]
        return attack_loss, prompt_logits, logits

@dataclasses.dataclass
class InternalObjectiveCosine(Objective):
    attack_layer: int = 20
    direction_vector: torch.Tensor = None

    def setup(self, new_victim):
        assert self.attack_mult > 0
        out = super().setup(new_victim, skip_ft_probs=True)
        out.attack_layers = [out.attack_layer]
        return out

    def calc_resid(self, *, attack_layers, **model_kwargs):
        resid = dict()

        def cache_resid(module, input, output, layer):
            resid[layer] = input[0].to(torch.float32)

        model = self.victim.base_model 
        hooks = []
        for L in attack_layers:
            hooks.append(
                (
                    model.model.layers[L - 1],
                    partial(cache_resid, layer=L),
                )
            )
        with contextlib.ExitStack() as stack:
            stack.enter_context(add_fwd_hooks(hooks))
            if model_kwargs.get("inputs_embeds", None) is None:
                model_out = model(**model_kwargs)
            else:
                del model_kwargs["input_ids"]
                model_out = model(**model_kwargs)
        return (
            torch.cat([resid[L][:, None] for L in attack_layers], dim=1),
            model_out,
        )

    def batch_attack(
        self,
        *,
        task_data: TaskData,
        prompt_ids,
        prompt_embeds,
        prompt_attention_mask,
        kv_cache=None,
        past_logits=None,
    ):
        n_batch = prompt_attention_mask.shape[0]

        model_kwargs = self.prep_args_for_model(
            task_data,
            prompt_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            kv_cache=kv_cache,
            past_logits=past_logits,
        )

        resid, model_out = self.calc_resid(
            attack_layers=self.attack_layers, **model_kwargs
        )
        if kv_cache is None:
            logits = model_out.logits
        else:
            logits = torch.cat((past_logits, model_out.logits), dim=1)

        resid = resid[:, :, -task_data.match_ids.shape[0] - 1 :]

        vocab_size = logits.shape[-1]
        force_logits = logits[:, -task_data.force_match_ids.shape[0] - 1 :]
        force_loss = (
            torch.nn.functional.cross_entropy(
                force_logits[:, task_data.force_slice].reshape(-1, vocab_size),
                task_data.force_ids[None].repeat(n_batch, 1).ravel(),
                reduction="none",
            )
            .reshape((n_batch, -1))
            .mean(dim=-1)
        )
        force_loss = torch.clamp(force_loss, min=-np.log(self.p_threshold))
        
        cos_sim = torch.nn.functional.cosine_similarity(resid, task_data.ft_resid, dim=-1)

        internal_loss = 3 * (cos_sim.reshape(n_batch, -1).mean(dim=-1))
        attack_loss = force_loss + internal_loss
        best_idx = attack_loss.argmin()
        logger.info(
            f"internal loss components: {force_loss[best_idx]:.3f} {internal_loss[best_idx]:.3f}"
        )

        prompt_logits = logits[:, : -task_data.force_match_ids.shape[0]]
        return attack_loss, prompt_logits, logits