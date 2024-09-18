import copy

import torch

from .victim import Attack, AttackSet


def delete(*, attack: AttackSet, tokenizer, n):
    parts = attack.attacks[0].parts
    part_ids = attack.attacks[0].get_part_ids(tokenizer)

    n_tokens_per_part = torch.tensor([p_ids.shape[0] for p_ids in part_ids])
    n_tokens = sum(n_tokens_per_part)
    part_ends = torch.cumsum(n_tokens_per_part, dim=0)
    part_starts = part_ends - n_tokens_per_part

    n = min(n, n_tokens)
    pos_probs = torch.full((n_tokens,), 1.0 / n_tokens)
    delete_pos = torch.multinomial(pos_probs, n, replacement=False).sort().values
    deleted_part = (
        (delete_pos[:, None] < part_ends[None, :]).to(torch.long).argmax(dim=-1)
    )

    part_delete_pos = delete_pos - part_starts[deleted_part]

    new_attacks = []
    for k in range(n):
        which_part = deleted_part[k]
        mask = torch.ones_like(part_ids[which_part], dtype=torch.bool)
        mask[part_delete_pos[k]] = 0
        new_part_ids = part_ids[which_part][mask]
        new_parts = copy.copy(parts)
        new_parts[which_part] = tokenizer.decode(new_part_ids)
        new_attacks.append(Attack(parts=new_parts))
    return new_attacks


def swap(*, attack: AttackSet, candidates, tokenizer, k1):
    parts = attack.attacks[0].parts
    part_ids = attack.attacks[0].get_part_ids(tokenizer)

    n_tokens_per_part = torch.tensor([p_ids.shape[0] for p_ids in part_ids])
    n_tokens = sum(n_tokens_per_part)
    part_ends = torch.cumsum(n_tokens_per_part, dim=0)
    part_starts = part_ends - n_tokens_per_part

    k2 = candidates[0].shape[1]
    if n_tokens * k2 < k1:
        swap_pos = torch.arange(n_tokens)
        candidate_idx = torch.arange(k2).repeat(swap_pos.shape[0])
        swap_pos = swap_pos.repeat(k2)
    else:
        swap_pos = torch.randint(0, n_tokens, (k1,)).sort().values
        candidate_idx = torch.randint(0, k2, (k1,))
    swapped_part = (
        (swap_pos[:, None] < part_ends[None, :]).to(torch.long).argmax(dim=-1)
    )

    new_attacks = []
    for k in range(swap_pos.shape[0]):
        which_part = swapped_part[k].item()
        part_swap_pos = swap_pos[k] - part_starts[which_part]
        new_part_ids = part_ids[which_part].clone()
        new_part_ids[part_swap_pos] = candidates[which_part][
            part_swap_pos, candidate_idx[k]
        ]
        new_parts = copy.copy(parts)
        new_parts[which_part] = tokenizer.decode(new_part_ids)
        new_attacks.append(Attack(parts=new_parts))
    return new_attacks


def insert(*, attack, candidates, tokenizer, k1, only_edges: bool = False):
    parts = attack.attacks[0].parts
    part_ids = attack.attacks[0].get_part_ids(tokenizer)

    n_tokens_per_part = torch.tensor([p_ids.shape[0] for p_ids in part_ids])
    elongated_n_tokens = sum(n_tokens_per_part + 1)
    elongated_part_ends = torch.cumsum(n_tokens_per_part + 1, dim=0)
    elongated_part_starts = elongated_part_ends - n_tokens_per_part - 1

    pos_probs = torch.zeros(elongated_n_tokens)
    if only_edges:
        for i in range(len(part_ids)):
            pos_probs[elongated_part_ends[i] - 1] = 1.0
    else:
        pos_probs[:] = 1.0

    k2 = candidates[0].shape[1]
    if torch.where(pos_probs > 0)[0].shape[0] * k2 < k1:
        insert_pos = torch.where(pos_probs > 0)[0]
        candidate_idx = torch.arange(k2).repeat(insert_pos.shape[0])
        insert_pos = insert_pos.repeat(k2)
    else:
        pos_probs /= pos_probs.sum()
        insert_pos = torch.multinomial(pos_probs, k1, replacement=True).sort().values
        candidate_idx = torch.randint(0, k2, (k1,))

    inserted_part = (
        (insert_pos[:, None] < elongated_part_ends[None, :])
        .to(torch.long)
        .argmax(dim=-1)
    )

    new_attacks = []
    for k in range(insert_pos.shape[0]):
        which_part = inserted_part[k].item()
        part_insert_pos = insert_pos[k] - elongated_part_starts[which_part]
        old_part_ids = part_ids[which_part].clone()
        new_part_ids = torch.empty(old_part_ids.shape[0] + 1, dtype=torch.long)
        new_part_ids[:part_insert_pos] = old_part_ids[:part_insert_pos]
        new_part_ids[part_insert_pos] = candidates[which_part][
            part_insert_pos, candidate_idx[k]
        ]
        new_part_ids[part_insert_pos + 1 :] = old_part_ids[part_insert_pos:]
        new_parts = copy.copy(parts)
        new_parts[which_part] = tokenizer.decode(new_part_ids)
        new_attacks.append(Attack(parts=new_parts))
    return new_attacks


def buffer_retain(
    buffer: AttackSet,
    candidate_eval: AttackSet,
    buffer_size: int,
    needs_candidates: bool,
):
    all_attacks = buffer.subset(range(1, len(buffer.attacks))).cat(candidate_eval)
    buffer_size = min(buffer_size, all_attacks.loss.shape[0])
    keep_idxs = all_attacks.loss.argsort()[:buffer_size]
    return all_attacks.subset(keep_idxs)
