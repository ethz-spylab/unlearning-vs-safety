import copy
import dataclasses
import logging
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import torch

from src.enhanced_gcg.flrt_repo.flrt.victim import (
    Attack,
    AttackSet,
    Victim,
    combine_evaluations,
    count_parts_in_template,
    estimate_batch_size,
    find_subseq_pos,
)

logger = logging.getLogger()


class TaskData:
    def __init__(self, fp, task: str, skip_ft_probs=False):
        self.task = task
        self.force = fp.force
        if self.force is None:
            if fp.force_template is not None:
                self.force = fp.force_template(task=task)
            else:
                self.force = fp.victim.cfg.first_token

        self.force_ids = fp.tokenizer.encode(
            self.force, return_tensors="pt", add_special_tokens=False
        )[0].to(torch.long)

        # hack to make the 2 new classes compatible
        from src.enhanced_gcg.flrt_repo.flrt.internal import InternalObjectiveOriginalModel, InternalObjectiveCosine
        if isinstance(fp, InternalObjectiveOriginalModel) \
                or isinstance(fp, InternalObjectiveCosine):
            fp.victim.ft_model = fp.original_model
            
        if fp.victim.ft_model is not None:
            n_parts = count_parts_in_template(fp.ft_template, fp.victim)
            ft_template = fp.ft_template
            if fp.ft_template is None:
                ft_template = fp.template
            self.ft_prompt = ft_template(
                Attack(parts=["" for _ in range(n_parts)]),
                task,
                self.force,
                fp.victim,
            )
            ft_input_ids = torch.cat(
                (
                    fp.tokenizer.encode(
                        self.ft_prompt, add_special_tokens=False, return_tensors="pt"
                    )[0],
                    self.force_ids,
                )
            )

            self.n_force = fp.n_force
            if self.n_force > self.force_ids.shape[0]:
                self.force_ids = fp.victim.ft_model.generate(
                    ft_input_ids.unsqueeze(0),
                    max_new_tokens=self.n_force - self.force_ids.shape[0],
                    min_new_tokens=self.n_force - self.force_ids.shape[0],
                )[0, -self.n_force :]
            else:
                self.n_force = self.force_ids.shape[0]

            self.ft_input_ids = torch.cat(
                (
                    fp.tokenizer.encode(
                        self.ft_prompt, add_special_tokens=False, return_tensors="pt"
                    )[0],
                    self.force_ids,
                )
            )
            self.n_match = fp.n_match
            if self.n_match > 0:
                self.match_ids = fp.victim.ft_model.generate(
                    self.ft_input_ids.unsqueeze(0),
                    max_new_tokens=self.n_match,
                    min_new_tokens=self.n_match,
                )[0, self.ft_input_ids.shape[0] :]
            else:
                self.match_ids = torch.tensor([], dtype=torch.long)
        else:
            self.n_match = 0
            self.match_ids = torch.tensor([], dtype=torch.long)

        self.force_match_ids = torch.cat((self.force_ids, self.match_ids))
        self.force_match_tok = fp.tokenizer.batch_decode(self.force_match_ids)

        self.force_match_embeds = fp.embeddings(self.force_match_ids).detach()
        self.force_match = fp.tokenizer.decode(self.force_match_ids)
        self.force_slice = slice(
            -self.force_match_ids.shape[0] - 1, -self.match_ids.shape[0] - 1
        )

        if self.n_match > 0 and not skip_ft_probs:
            self.match_slice = slice(-(self.match_ids.shape[0] + 1), None)
            ft_logits = fp.victim.ft_model(
                torch.cat((self.ft_input_ids, self.match_ids)).unsqueeze(0)
            ).logits[0, self.match_slice]
            self.ft_probs = torch.softmax(ft_logits, dim=-1)

        if fp.attack_mult > 0:
            logger.info(f"MODEL: {fp.victim.name}")
            logger.info(f"TASK: {self.task}")
            logger.info(f"PROMPT: {self.ft_prompt}")
            logger.info(fp.tokenizer.decode(self.force_match_ids))


@dataclasses.dataclass
class Objective:
    victim: Union[Victim, str]
    tasks: str
    force: str = None
    attack_mult: float = 1.0
    fluency_mult: float = 0.0
    repetition_mult: float = 0.0
    repetition_exponent: float = 1.5
    p_threshold: float = 0.6
    n_force: int = 6
    n_match: int = 32
    template: Callable = None
    ft_template: Callable = None
    force_template: Callable = None
    task_data: List[TaskData] = None
    use_prefix_evaluate: bool = False

    def __post_init__(self):
        from .templates import default_template, task_only_template

        if isinstance(self.victim, str):
            self.victim = Victim(name=self.victim)
        if self.template is None:
            self.template = default_template
        if self.ft_template is None:
            self.ft_template = task_only_template
        if len(self.tasks) == 1 and self.use_prefix_evaluate:
            logger.warning(
                "Turning off prefix evaluation because there is only one task."
            )
            self.use_prefix_evaluate = False

    @property
    def tokenizer(self):
        return self.victim.tokenizer

    def serialize(self):
        s = copy.copy(self)
        s.template = s.serialize_template(s.template)
        s.ft_template = s.serialize_template(s.ft_template)
        s.force_template = s.serialize_force_template()
        s.victim = s.victim.serialize()
        return dataclasses.asdict(s)

    def serialize_template(self, tmpl):
        return tmpl(
            task="{task}",
            attack=Attack(
                parts=[
                    ("{attack.parts[" + str(i) + "]}")
                    for i in range(count_parts_in_template(tmpl, self.victim))
                ]
            ),
            injection="{injection}",
            victim=self.victim,
        )

    def serialize_force_template(self):
        if self.force_template is not None:
            return self.force_template(task="{task}")
        return None

    def setup(self, new_victim: Victim, skip_ft_probs: bool = False):
        out = copy.copy(self)
        out.victim = new_victim
        out.embeddings = out.victim.base_model.get_input_embeddings()

        out.fake_str = "INSERTFAKEPARTHURRAYJAILBREAK"
        out.fake_p_ids = out.tokenizer.encode(
            out.fake_str,
            return_tensors="pt",
            add_special_tokens=False,
        )[0]

        out.task_data = [TaskData(out, t, skip_ft_probs) for t in self.tasks]
        return out

    def gcg_candidates(self, attack: Attack, topk: int):
        task_grads = []
        for task_data in self.task_data:
            with torch.enable_grad():
                prompt = self.template(
                    attack, task_data.task, task_data.force_match, self.victim
                )
                input_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                embed = self.victim.base_model.get_input_embeddings()
                one_hot = torch.nn.functional.one_hot(
                    input_ids, num_classes=embed.num_embeddings
                ).to(embed.weight.dtype)
                one_hot.requires_grad = True
                inputs_embeds = torch.matmul(one_hot, embed.weight)

                attack_loss, prompt_logits, logits = self.batch_attack(
                    task_data=task_data,
                    prompt_ids=input_ids,
                    prompt_embeds=inputs_embeds,
                    prompt_attention_mask=attention_mask,
                    kv_cache=None,
                    past_logits=None,
                )

                e = self.batch_evaluate(
                    task_data=task_data,
                    attacks=[attack],
                    prompt_ids=input_ids,
                    attention_mask=attention_mask,
                    attack_loss=attack_loss,
                    logits=logits,
                    prompt_logits=prompt_logits,
                    return_convergence_report=False,
                )
                e.loss.sum().backward()

                part_ids = attack.get_part_ids(self.tokenizer)
                task_grads.append(
                    [
                        one_hot.grad[0, start : start + len(p_ids)].clone()
                        for (start, p_ids) in zip(e.starts[0], part_ids)
                    ]
                )
        n_tasks = len(self.task_data)
        n_parts = len(task_grads[0])
        grads = [
            torch.stack([task_grads[i][j] for i in range(n_tasks)], dim=0).mean(dim=0)
            for j in range(n_parts)
        ]
        candidates = [(-grads[i]).topk(k=topk, dim=-1).indices for i in range(n_parts)]
        return candidates

    def beast_candidates(self, attack: Attack, k2: int):
        task_data = self.task_data[torch.randint(len(self.task_data), (1,)).item()]
        prompt = self.template(
            attack, task_data.task, task_data.force_match, self.victim
        )
        prompt_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        model_out = self.victim.base_model(prompt_ids, use_cache=False)
        starts, _, part_ids = self.find_part_starts(task_data, [attack], prompt_ids)
        token_probs = [
            model_out.logits[0, start - 1 : start + len(p_ids)].softmax(dim=-1)
            for (start, p_ids) in zip(starts[0], part_ids[0])
        ]
        candidates = [
            torch.multinomial(tp, k2, replacement=False) for tp in token_probs
        ]
        return candidates

    def evaluate(self, attacks: List[Attack]):
        if self.use_prefix_evaluate:
            return self.prefix_evaluate(attacks)
        else:
            return self.simple_evaluate(attacks)

    def simple_evaluate(self, attacks: List[Attack]):
        e_multi = []
        for task_data in self.task_data:
            prompts = [
                self.template(t, task_data.task, task_data.force_match, self.victim)
                for t in attacks
            ]
            tokenized = self.tokenizer(
                prompts, add_special_tokens=False, return_tensors="pt", padding=True
            )

            batch_size = estimate_batch_size(
                tokenized["input_ids"].shape[1],
                self.embeddings.num_embeddings,
            )

            e = None
            df = None
            for i in range(0, len(attacks), batch_size):
                if self.attack_mult > 0:
                    attack_loss, prompt_logits, logits = self.batch_attack(
                        task_data=task_data,
                        prompt_ids=tokenized["input_ids"][i : i + batch_size],
                        prompt_embeds=None,
                        prompt_attention_mask=tokenized["attention_mask"][
                            i : i + batch_size
                        ],
                    )
                else:
                    attack_loss, prompt_logits, logits = self.batch_nonattack(
                        task_data=task_data,
                        prompt_ids=tokenized["input_ids"][i : i + batch_size],
                        prompt_embeds=None,
                        prompt_attention_mask=tokenized["attention_mask"][
                            i : i + batch_size
                        ],
                    )
                e_new = self.batch_evaluate(
                    task_data=task_data,
                    attacks=attacks[i : i + batch_size],
                    prompt_ids=tokenized["input_ids"][i : i + batch_size],
                    attention_mask=tokenized["attention_mask"][i : i + batch_size],
                    attack_loss=attack_loss,
                    logits=logits,
                    prompt_logits=prompt_logits,
                )
                if e is None:
                    e = e_new
                    df = e.convergence_reports[0]
                else:
                    e = e.cat(e_new)
                    if e_new.loss.min() < e.loss.min():
                        df = e_new.convergence_reports[0]
            e.convergence_reports = [df]
            e_multi.append(e)

        return combine_evaluations([self] * len(e_multi), e_multi)

    def prefix_evaluate(self, attacks: List[Attack]):
        shared_prefixes = []
        suffixes = []
        suffix_masks = []
        self.tokenizer.padding_side = "right"
        n_tasks = len(self.task_data)
        for t in attacks:
            prompts = [
                self.template(t, task_data.task, task_data.force_match, self.victim)
                for task_data in self.task_data
            ]
            tokenized = self.tokenizer(
                prompts, add_special_tokens=False, return_tensors="pt", padding=True
            )
            token_shared = (
                tokenized["input_ids"][0, None, :] == tokenized["input_ids"]
            ).all(dim=0)
            shared_end = (
                torch.cumsum(token_shared, 0)
                == torch.arange(1, token_shared.shape[0] + 1)
            ).nonzero(as_tuple=True)[0].max() + 1
            shared_prefix = tokenized["input_ids"][0, :shared_end]
            shared_prefixes.append(shared_prefix)
            suffixes.append(tokenized["input_ids"][:, shared_end:])
            suffix_masks.append(tokenized["attention_mask"][:, shared_end:])
        self.tokenizer.padding_side = "left"
        shared_prefixes = self.tokenizer.pad(
            {"input_ids": [x.tolist() for x in shared_prefixes]}, return_tensors="pt"
        )

        batch_size = (
            estimate_batch_size(
                tokenized["input_ids"].shape[1],
                self.embeddings.num_embeddings,
            )
            // 8
        )

        e_multi = dict()
        for i in range(0, len(attacks), batch_size):
            model_out = self.victim.base_model(
                input_ids=shared_prefixes["input_ids"][i : i + batch_size],
                attention_mask=shared_prefixes["attention_mask"][i : i + batch_size],
                use_cache=True,
            )
            batch_suffixes = self.tokenizer.pad(
                {
                    "input_ids": sum(
                        [
                            [
                                suffixes[j][k, suffix_masks[j][k].to(torch.bool)]
                                for k in range(n_tasks)
                            ]
                            for j in range(i, min(i + batch_size, len(attacks)))
                        ],
                        [],
                    )
                },
                return_tensors="pt",
            )
            batch_input_ids = batch_suffixes["input_ids"]
            batch_attention_mask = torch.cat(
                (
                    shared_prefixes["attention_mask"][
                        i : i + batch_size
                    ].repeat_interleave(n_tasks, dim=0),
                    batch_suffixes["attention_mask"],
                ),
                dim=1,
            )
            for k, task_data in enumerate(self.task_data):
                if self.attack_mult > 0:
                    # tokenized2 = self.tokenizer(
                    #     [
                    #         self.template(
                    #             t, task_data.task, task_data.force_match, self.victim
                    #         )
                    #         for t in attacks[i : i + batch_size]
                    #     ],
                    #     add_special_tokens=False,
                    #     return_tensors="pt",
                    #     padding=True,
                    # )
                    attack_loss, prompt_logits, logits = self.batch_attack(
                        task_data=task_data,
                        prompt_ids=batch_input_ids[k::n_tasks],
                        prompt_embeds=None,
                        prompt_attention_mask=batch_attention_mask[k::n_tasks],
                        kv_cache=model_out.past_key_values,
                        past_logits=model_out.logits,
                        # full_prompt_ids=tokenized2["input_ids"],
                        # full_attention_mask=tokenized2["attention_mask"],
                    )
                else:
                    attack_loss, prompt_logits, logits = self.batch_nonattack(
                        task_data=task_data,
                        prompt_ids=batch_input_ids[k::n_tasks],
                        prompt_embeds=None,
                        prompt_attention_mask=batch_attention_mask[k::n_tasks],
                        kv_cache=model_out.past_key_values,
                        past_logits=model_out.logits,
                    )
                e = self.batch_evaluate(
                    task_data=task_data,
                    attacks=attacks[i : i + batch_size],
                    prompt_ids=torch.cat(
                        (
                            shared_prefixes["input_ids"][i : i + batch_size],
                            batch_input_ids[k::n_tasks],
                        ),
                        dim=1,
                    ),
                    attention_mask=batch_attention_mask[k::n_tasks],
                    attack_loss=attack_loss,
                    logits=logits,
                    prompt_logits=prompt_logits,
                )
                if task_data.task not in e_multi:
                    e_multi[task_data.task] = e
                else:
                    e_multi[task_data.task] = e_multi[task_data.task].cat(e)

        return combine_evaluations([self] * len(e_multi), list(e_multi.values()))

    def prep_args_for_model(
        self,
        task_data,
        prompt_ids,
        prompt_embeds,
        prompt_attention_mask,
        kv_cache=None,
        past_logits=None,
    ):
        n_batch = prompt_attention_mask.shape[0]

        force_match_expand = task_data.force_match_ids[None].repeat(n_batch, 1)
        prompt_force_ids = torch.cat((prompt_ids, force_match_expand), dim=1)
        attention_mask = torch.cat(
            (
                prompt_attention_mask,
                torch.ones_like(force_match_expand, dtype=torch.bool),
            ),
            dim=1,
        )
        if prompt_embeds is None:
            inputs_embeds = None
            input_ids = prompt_force_ids
        else:
            input_ids = None
            inputs_embeds = torch.cat(
                (
                    prompt_embeds,
                    task_data.force_match_embeds[None].repeat(n_batch, 1, 1),
                ),
                dim=1,
            )

        position_ids = (torch.cumsum(attention_mask, dim=-1) - 1)[
            :, -prompt_force_ids.shape[1] :
        ]
        return dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=False if kv_cache is None else True,
        )

    def batch_attack(
        self,
        *,
        task_data,
        prompt_ids,
        prompt_embeds,
        prompt_attention_mask,
        kv_cache=None,
        past_logits=None,
        # TODO: remove, debugging
        # full_prompt_ids=None,
        # full_attention_mask=None,
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
        # NOTE: substantial memory optimization potential from never
        # materializing the logits tensors and using fused kernels to calculate
        # attack and fluency XE
        model_out = self.victim.base_model(**model_kwargs)

        # if full_prompt_ids is not None:
        #     prompt_force_ids2 = torch.cat((full_prompt_ids, force_match_expand), dim=1)
        #     attention_mask2 = torch.cat(
        #         (
        #             full_attention_mask,
        #             torch.ones_like(force_match_expand, dtype=torch.bool),
        #         ),
        #         dim=1,
        #     )
        #     model_out2 = self.victim.base_model(
        #         input_ids=prompt_force_ids2,
        #         attention_mask=attention_mask2,
        #         use_cache=False if kv_cache is None else True,
        #     )
        #     past_subset = past_logits[
        #         0, attention_mask[0, : past_logits[0].shape[0]].to(torch.bool)
        #     ]
        #     logit_subset = model_out.logits[
        #         0, attention_mask[0, past_logits[0].shape[0] :].to(torch.bool)
        #     ]
        #     logit2_subset = model_out2.logits[0, attention_mask2[0].to(torch.bool)]
        #     print("whoa")
        #     nprior = attention_mask[0, : past_logits[0].shape[0]].sum()
        #     diff_early = past_subset - logit2_subset[:nprior]
        #     diff_later = logit_subset - logit2_subset[nprior:]
        #     print((diff_early**2).mean(dim=-1))
        #     print((diff_later**2).mean(dim=-1))
        #     print(diff_later.max())
        #     if diff_later.max() < 1e-5:
        #         print("WHOA")

        if kv_cache is None:
            logits = model_out.logits
        else:
            logits = torch.cat((past_logits, model_out.logits), dim=1)

        vocab_size = logits.shape[-1]
        force_logits = logits[:, -(task_data.force_match_ids.shape[0] + 1) :]
        attack_xe = torch.empty(
            (
                n_batch,
                task_data.force_ids.shape[0]
                + task_data.match_ids.shape[0]
                + (1 if task_data.n_match > 0 else 0),
            )
        )
        attack_xe[
            :, : task_data.force_ids.shape[0]
        ] = torch.nn.functional.cross_entropy(
            force_logits[:, task_data.force_slice].reshape(-1, vocab_size),
            task_data.force_ids[None].repeat(n_batch, 1).ravel(),
            reduction="none",
        ).reshape(
            (n_batch, -1)
        )
        if task_data.n_match > 0:
            # TODO: this line approximately doubles memory consumption
            attack_xe[
                :, task_data.force_ids.shape[0] :
            ] = torch.nn.functional.cross_entropy(
                force_logits[:, task_data.match_slice].reshape(-1, vocab_size),
                task_data.ft_probs[None].repeat(n_batch, 1, 1).reshape(-1, vocab_size),
                reduction="none",
            ).reshape(
                (n_batch, -1)
            )

        attack_xe_mod = torch.clamp(attack_xe, min=-np.log(self.p_threshold))
        multipliers = torch.ones(attack_xe_mod.shape[1])
        multipliers[0] *= 1.5
        multipliers[: task_data.n_force] *= 3
        multipliers = multipliers * attack_xe_mod.shape[1] / multipliers.sum()
        attack_xe_mod *= multipliers[None]
        attack_loss = attack_xe_mod.mean(dim=-1)

        prompt_logits = logits[:, : -task_data.force_match_ids.shape[0]]
        return attack_loss, prompt_logits, logits

    def batch_nonattack(
        self,
        *,
        task_data,
        prompt_ids,
        prompt_embeds,
        prompt_attention_mask,
        kv_cache=None,
        past_logits=None,
    ):
        attack_loss = torch.zeros(prompt_attention_mask.shape[0])
        model_out = self.victim.base_model(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            past_key_values=kv_cache,
            use_cache=False if kv_cache is None else True,
        )
        if kv_cache is None:
            logits = model_out.logits
        else:
            logits = torch.cat((past_logits, model_out.logits), dim=1)
        return attack_loss, logits, logits

    def batch_evaluate(
        self,
        *,
        task_data: TaskData,
        attacks: List[Attack],
        prompt_ids,
        attention_mask,
        attack_loss,
        logits,
        prompt_logits,
        return_convergence_report: bool = True,
    ):
        n_batch = len(attacks)
        vocab_size = logits.shape[-1]
        if self.fluency_mult > 0:
            labels = prompt_ids[:, 1:].clone()
            labels[~(attention_mask[:, 1:].to(torch.bool))] = -100
            xentropy = (
                torch.nn.functional.cross_entropy(
                    prompt_logits[:, :-1].reshape(-1, vocab_size),
                    labels.ravel(),
                    reduction="none",
                )
                .reshape(n_batch, -1)
                .mean(dim=-1)
            )
        else:
            xentropy = torch.zeros(n_batch)

        if self.repetition_mult > 0:
            counts = torch.zeros((n_batch, vocab_size), dtype=torch.int32)
            counts.scatter_add_(
                1, prompt_ids, torch.ones_like(prompt_ids, dtype=torch.int32)
            )
            repetition = (
                torch.clamp((counts - 1), min=0).to(torch.float)
                ** self.repetition_exponent
            ).sum(dim=-1) / attention_mask.sum(dim=-1)
        else:
            repetition = torch.zeros(n_batch)

        starts, illegal_penalty, part_ids = self.find_part_starts(
            task_data, attacks, prompt_ids
        )

        attack_loss += illegal_penalty
        xentropy += illegal_penalty
        repetition += illegal_penalty

        loss = (
            self.attack_mult * attack_loss
            + self.fluency_mult * xentropy
            + self.repetition_mult * repetition
        )

        if self.attack_mult > 0 and return_convergence_report:
            # Termination criterion: Is the force attack_loss the most likely token in
            # each position by a specified probability margin.
            best_idx = loss.argmin()
            n_force = task_data.force_match_ids.shape[0]
            force_logits = logits[:, -n_force - 1 :]
            p = torch.softmax(force_logits[best_idx], dim=-1)
            p_top2 = p[:n_force].topk(k=2, dim=-1)
            p_0 = p_top2.values[:, 0]
            p_1 = p_top2.values[:, 1]
            ids_0 = p_top2.indices[:, 0]
            ids_1 = p_top2.indices[:, 1]

            # Report on the most likely tokens
            report = (
                self.victim.cfg.short_name + " " + task_data.task,
                pd.DataFrame(
                    dict(
                        tok_force=task_data.force_match_tok,
                        p_force=p[torch.arange(n_force), task_data.force_match_ids]
                        .cpu()
                        .numpy(),
                        tok_0=self.tokenizer.batch_decode(ids_0),
                        p_0=p_0.cpu().numpy(),
                        tok_1=self.tokenizer.batch_decode(ids_1),
                        p_1=p_1.cpu().numpy(),
                        flagged=(task_data.force_match_ids != ids_0).cpu().numpy(),
                    )
                ),
            )
        else:
            report = None

        return AttackSet(
            attacks=attacks,
            loss=loss,
            attack_loss=attack_loss,
            xentropy=xentropy,
            repetition=repetition,
            starts=starts,
            convergence_reports=[report],
        )

    def find_part_starts(self, task_data, attacks, prompt_ids):
        illegal_penalty = torch.zeros(len(attacks))
        starts = [[] for k in range(len(attacks))]
        part_ids = []
        # NOTE: vectorizing this is feasible but not worth the effort at the
        # moment. chatgpt conversation where we figured it out together... i
        # think the solution is slightly wrong but correct in spirit.
        # https://chatgpt.com/share/9cd38bd2-ffbd-43fe-a302-01fff812b64b
        # vectorizing this would speed up the code by about 10%.
        for k, a in enumerate(attacks):
            part_ids.append(a.get_part_ids(self.tokenizer))
            for R, p_ids in enumerate(part_ids[k]):
                illegal_penalty[k] += (
                    torch.isin(p_ids, self.victim.banned_tokens).any().to(torch.float)
                    * 100
                )

                done = False
                if len(p_ids) > 4:
                    idxs = find_subseq_pos(prompt_ids[k], p_ids)
                    if len(idxs) == 1:
                        starts[k].append(idxs[0])
                        done = True
                    else:
                        illegal_penalty[k] += 1.0

                if not done:
                    t_fake = Attack(parts=a.parts.copy())
                    t_fake.parts[R] = self.fake_str
                    fake_prompt_ids = self.tokenizer.encode(
                        self.template(
                            t_fake, task_data.task, task_data.force_match, self.victim
                        ),
                        add_special_tokens=False,
                        return_tensors="pt",
                    )[0]
                    for cut in range(2):
                        if cut > 1:
                            illegal_penalty[k] += 1.0
                        search_ids = self.fake_p_ids
                        if cut > 0:
                            search_ids = search_ids[cut:-cut]
                        idxs = find_subseq_pos(fake_prompt_ids, search_ids)
                        if len(idxs) > 0:
                            break
                    if len(idxs) == 0:
                        raise ValueError(
                            f"Part identification failed! (model={self.victim.name}, task={repr(task_data.task)}, attack={a})"
                        )
                    starts[k].append(idxs[0] - cut)
        return starts, illegal_penalty, part_ids
