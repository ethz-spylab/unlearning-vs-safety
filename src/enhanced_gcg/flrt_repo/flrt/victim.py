import copy
import dataclasses
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import transformers

import src.enhanced_gcg.flrt_repo.flrt.util as util


@dataclasses.dataclass(slots=True)
class Attack:
    parts: List[str]

    def get_part_ids(self, tokenizer):
        return [
            tokenizer.encode(p, return_tensors="pt", add_special_tokens=False)[0]
            for p in self.parts
        ]

    def serialize(self):
        return dict(parts=self.parts)


@dataclasses.dataclass(slots=True)
class AttackSet:
    attacks: List[Attack] = None
    loss: torch.Tensor = None
    attack_loss: torch.Tensor = None
    xentropy: torch.Tensor = None
    repetition: torch.Tensor = None
    starts: List[List[int]] = None
    convergence_reports: List[pd.DataFrame] = None

    def serialize(self):
        s = copy.copy(self)
        s.attacks = [t.serialize() for t in s.attacks]
        s.loss = s.loss.cpu().numpy()
        s.attack_loss = s.attack_loss.cpu().numpy()
        s.xentropy = s.xentropy.cpu().numpy()
        s.repetition = s.repetition.cpu().numpy()
        return dataclasses.asdict(s)

    def cat(self, other):
        if self.loss.shape[0] == 0:
            return other
        if other.loss.shape[0] == 0:
            return self

        return AttackSet(
            attacks=self.attacks + other.attacks,
            loss=torch.cat([self.loss, other.loss]),
            attack_loss=torch.cat([self.attack_loss, other.attack_loss]),
            xentropy=torch.cat([self.xentropy, other.xentropy]),
            repetition=torch.cat([self.repetition, other.repetition]),
            starts=(self.starts + other.starts if self.starts is not None else None),
            convergence_reports=(
                self.convergence_reports
                if self.loss.min() < other.loss.min()
                else other.convergence_reports
            ),
        )

    def subset(self, indices):
        return AttackSet(
            attacks=[self.attacks[i] for i in indices],
            loss=self.loss[indices],
            attack_loss=self.attack_loss[indices],
            xentropy=self.xentropy[indices],
            repetition=self.repetition[indices],
            starts=(
                [self.starts[i] for i in indices] if self.starts is not None else None
            ),
        )


def combine_evaluations(objectives, evaluations):
    n_attack = max(len([0 for obj in objectives if obj.attack_mult > 0]), 1)
    n_fluency = max(len([0 for obj in objectives if obj.fluency_mult > 0]), 1)
    n_repetition = max(len([0 for obj in objectives if obj.repetition_mult > 0]), 1)

    attack_loss = sum([e.attack_loss for e in evaluations]) / n_attack
    xentropy = sum([e.xentropy for e in evaluations]) / n_fluency
    repetition = sum([e.repetition for e in evaluations]) / n_repetition

    loss = sum(
        [
            obj.attack_mult * e.attack_loss / n_attack
            + obj.fluency_mult * e.xentropy / n_fluency
            + obj.repetition_mult * e.repetition / n_repetition
            for obj, e in zip(objectives, evaluations)
        ]
    )
    return AttackSet(
        attacks=evaluations[0].attacks,
        loss=loss,
        attack_loss=attack_loss,
        xentropy=xentropy,
        repetition=repetition,
        convergence_reports=sum([e.convergence_reports for e in evaluations], []),
    )


def find_subseq_pos(seq1, seq2, start_search=0):
    return (
        seq1[torch.arange(seq1.shape[0]).unfold(0, seq2.shape[0], 1)]
        .eq(seq2)
        .all(dim=1)
        .nonzero()[:, 0]
        + start_search
    )


def random_attack(objectives, part_lens, tasks):
    attack = Attack(parts=["" for _ in part_lens])
    for i, L in enumerate(part_lens):
        n_tries = 10
        success = False
        for j in range(n_tries):
            if _random_part(objectives, attack, i, L, tasks):
                success = True
                break
            else:
                print("retrying initial attack")
        if not success:
            raise ValueError(f"Failed to find a legal attack for part {i}")
    return attack


def _random_part(objectives, attack, i, L, tasks):
    attack.parts[i] = objectives[0].tokenizer.decode(
        torch.randint(
            0,
            objectives[0].tokenizer.vocab_size,
            (L,),
        )
    )

    return check_and_repair_attack(attack, i, objectives, should_repair=True)


def count_parts_in_template(tmpl, victim):
    for i in range(15):
        try:
            tmpl(task="a", attack=Attack(parts=range(i)), injection="", victim=victim)
        except IndexError:
            continue
        else:
            return i
    raise ValueError("Failed to find number of parts")


def check_and_repair_attack(attack, part_idx, objectives, should_repair):
    for o in objectives:
        for tsk in o.tasks:
            v = o.victim
            prompt = o.template(attack, tsk, "", v)
            prompt_ids = v.tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )[0]
            part_ids = v.tokenizer.encode(
                attack.parts[part_idx], return_tensors="pt", add_special_tokens=False
            )[0]
            if torch.isin(part_ids, v.banned_tokens).any():
                return False
            idxs = find_subseq_pos(prompt_ids, part_ids)
            if len(idxs) == 1:
                continue

            if not should_repair:
                return False

            for trim in range(4):
                idxs_begin_trim = find_subseq_pos(prompt_ids, part_ids[trim:])
                if len(idxs_begin_trim) == 1:
                    attack.parts[part_idx] = v.tokenizer.decode(part_ids[trim:])
                    return check_and_repair_attack(
                        attack, part_idx, objectives, should_repair
                    )

            for trim in range(4):
                idxs_end_trim = find_subseq_pos(prompt_ids, part_ids[:-trim])
                if len(idxs_end_trim) == 1:
                    attack.parts[part_idx] = v.tokenizer.decode(part_ids[:-trim])
                    return check_and_repair_attack(
                        attack, part_idx, objectives, should_repair
                    )

            return False
    return True


def estimate_batch_size(n_tokens: int, n_embed: int, safety_factor: int = 4):
    unreserved_mem, total_mem = torch.cuda.mem_get_info()
    reserved_mem = torch.cuda.memory_reserved()
    used_mem = torch.cuda.memory_allocated()
    available_mem = unreserved_mem + reserved_mem - used_mem
    other_proc_mem = total_mem - unreserved_mem - reserved_mem
    if other_proc_mem > 2e9:
        print("Warning: other process GPU memory usage is high:", other_proc_mem)
    logit_mem_usage = n_tokens * n_embed * 4
    safety_factor = 4
    return 2 ** int(np.log2(available_mem / logit_mem_usage / safety_factor))


@dataclasses.dataclass(slots=True)
class Victim:
    name: str = None
    cfg: util.ModelConfig = None
    extra_banned_tokens: List[int] = None

    _tokenizer: transformers.PreTrainedTokenizer = None
    base_model: torch.nn.Module = None
    ft_model: torch.nn.Module = None
    banned_tokens: List[int] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.cfg.model_name
        if self.cfg is None:
            self.cfg = util.model_configs[self.name]
        if self.extra_banned_tokens is None:
            self.extra_banned_tokens = []

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.cfg.tokenizer_name is None:
                self.cfg.tokenizer_name = self.cfg.model_name
            self._tokenizer = util.load_tokenizer(self.cfg.tokenizer_name)
        return self._tokenizer

    def serialize(self):
        s = copy.copy(self)
        out = dataclasses.asdict(s)
        del out["base_model"]
        del out["ft_model"]
        del out["_tokenizer"]
        return out

    def setup(self, model=None, skip_ft_model: bool = False):
        if model is None:
            model = util.load_model(self.cfg.model_name)

        if self.cfg.peft_path is not None and not skip_ft_model:
            from peft import PeftModel

            ft_model = PeftModel.from_pretrained(
                model, os.path.join("output", "flrt-finetune", self.cfg.peft_path)
            )
            base_model = BaseModel(ft_model)
        else:
            ft_model = None
            base_model = model

        if base_model is not None:
            num_embeddings = base_model.get_input_embeddings().num_embeddings
        else:
            num_embeddings = self.tokenizer.vocab_size

        banned_tokens = self.cfg.banned_tokens
        if banned_tokens is None:
            banned_tokens = self.tokenizer.all_special_ids + list(
                range(self.tokenizer.vocab_size, num_embeddings)
            )
        banned_tokens = banned_tokens + self.extra_banned_tokens
        banned_tokens = torch.tensor(list(set(banned_tokens)), dtype=torch.long)
        kwargs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        kwargs.update(
            dict(
                base_model=base_model,
                ft_model=ft_model,
                banned_tokens=banned_tokens,
            )
        )
        return Victim(**kwargs)


class BaseModel:
    """
    Forwards all calls to the wrapped peft model with the adapter
    disabled.
    """

    def __init__(self, peft_model):
        self.peft_model = peft_model

    def __call__(self, *args, **kwargs):
        with self.peft_model.disable_adapter():
            return self.peft_model(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self.peft_model, name)
        if name == "peft_model":
            return attr

        if callable(attr):

            def wrapper(*args, **kwargs):
                with self.peft_model.disable_adapter():
                    return attr(*args, **kwargs)

            return wrapper
        return attr


def generate(
    model,
    tokenizer,
    prompts,
    force_ids=None,
    batch_size=16,
    **kwargs,
):
    tokenized = tokenizer(
        prompts, return_tensors="pt", add_special_tokens=False, padding=True
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    if force_ids is not None:
        n_batch = len(prompts)
        force_expand = force_ids[None].repeat(n_batch, 1)
        input_ids = torch.cat((input_ids, force_expand), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones_like(force_expand, dtype=torch.bool),
            ),
            dim=1,
        )
    all_ids = torch.cat(
        [
            util.hf_generate(
                model,
                input_ids[i : i + batch_size],
                attention_mask=attention_mask[i : i + batch_size],
                **kwargs,
            )
            for i in range(0, len(prompts), batch_size)
        ],
        dim=0,
    )
    assert (all_ids[:, : input_ids.shape[1]] == input_ids).all()
    generation_ids = all_ids[:, input_ids.shape[1] :]
    return dict(
        prompts=prompts,
        all_ids=all_ids,
        generation_ids=generation_ids,
        generations=tokenizer.batch_decode(generation_ids),
    )
