import copy
import dataclasses
import datetime
import json
import logging
import multiprocessing
import os
import pickle
import pprint
import random
import shutil
import sys
import time
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch

import src.enhanced_gcg.flrt_repo.flrt.judge as judge
import src.enhanced_gcg.flrt_repo.flrt.modal_defs as modal_defs
import src.enhanced_gcg.flrt_repo.flrt.operators as operators
import src.enhanced_gcg.flrt_repo.flrt.util as util
import src.enhanced_gcg.flrt_repo.flrt.victim as victim
from src.enhanced_gcg.flrt_repo.flrt.internal import InternalObjective, InternalObjectiveOriginalModel, InternalObjectiveCosine  # noqa
from src.enhanced_gcg.flrt_repo.flrt.objective import Objective
from src.enhanced_gcg.flrt_repo.flrt.victim import Attack, AttackSet, Victim  # noqa
from src.util.globals import OUTPUT_DIR

logger = logging.getLogger()


@dataclasses.dataclass(slots=True)
class Settings:
    end: int
    k1: int = 32
    k2: int = 16
    buffer_size: int = 8
    fluency_mult: float = 0.0
    repetition_mult: float = 0.0
    p_delete: float = 0.0
    p_insert: float = 0.0
    p_swap: float = 0.0
    p_edge: float = 0.0
    p_gcg: float = 0.0


@dataclasses.dataclass(slots=True)
class AttackConfig:
    config: str = ""

    # Output parameters
    project_name: str = "flrt"
    run_name: str = None
    s3_bucket: str = "caiplay"
    wandb_log: bool = True
    wandb_mode: str = "online"
    wandb_log_freq: int = 10
    generate_tokens: int = 64
    final_generate_tokens: int = 512
    checkpoint_freq: int = 100
    n_postprocess_select: int = 10

    # Objective functions!
    objectives: List[Objective] = None

    # Prompt
    load_checkpoint: str = None
    attack_parts: List[str] = None
    attack_part_lens: List[int] = None
    min_tokens: int = 1
    max_tokens: int = 1024
    start_tokens: int = 32
    token_length_ramp: int = None

    # Algorithm parameters
    schedule: List[Settings] = None
    iters: int = None
    runtime_limit: int = 10 * 60  # seconds
    seed: int = 0

    def serialize(self):
        s = copy.copy(self)
        s.objectives = [obj.serialize() for obj in s.objectives]
        return dataclasses.asdict(s)


def load_checkpoint(
    bucket, project_name, run_name, use_iter=None, download_if_exists=False
):
    import s3fs

    local_folder = os.path.join("output", project_name, run_name)
    s3_folder = os.path.join(bucket, project_name, run_name)
    fs = None
    if os.path.exists(local_folder):
        filenames = [f for f in os.listdir(local_folder)]
    else:
        fs = s3fs.S3FileSystem()
        filenames = [os.path.basename(f) for f in fs.ls(s3_folder)]

    if use_iter is None:
        use_iter = -1
        for f in filenames:
            if "iter" in f:
                use_iter = max(use_iter, int(f.split("iter")[1].split(".pkl")[0]))
    filename = f"iter{use_iter}.pkl"
    s3_filepath = os.path.join(s3_folder, filename)
    local_filepath = os.path.join(local_folder, filename)
    if not os.path.exists(local_filepath) or download_if_exists:
        if fs is None:
            fs = s3fs.S3FileSystem()
        fs.download(s3_filepath, local_filepath)
    with open(local_filepath, "rb") as f:
        return pickle.load(f)


def multi_evaluate(
    objectives: List[Objective],
    attacks: List[Attack],
):
    es = [o.evaluate(attacks) for o in objectives]
    return victim.combine_evaluations(objectives, es)


def attack(c: AttackConfig):
    pd.set_option("display.max_columns", 500)

    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_filename = f"log-{c.project_name}-{c.run_name}-{now}.txt"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    stdout_handler.setLevel(logging.INFO)

    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)

    try:
        output = _attack(c)
    except:
        logger.exception("Exception occurred during attack")
        raise
    finally:
        output_dir = os.path.join(OUTPUT_DIR, "enhanced_gcg", "logs", c.project_name, c.run_name)

        for h in logging.getLogger().handlers:
            h.flush()
            h.close()
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(log_path, f"{output_dir}/{log_filename}")

        if c.wandb_log:
            import wandb

            shutil.copytree("wandb/latest-run", f"{output_dir}/wandb")
            wandb.finish()
        return output


@torch.no_grad()
def _attack(c: AttackConfig):
    if c.wandb_log:
        import wandb

        wandb.require("core")

        assert c.run_name is not None
        wandb.init(
            reinit=True,
            project=c.project_name,
            name=c.run_name,
            mode=c.wandb_mode,
            config=c.serialize(),
        )

    logger.info("Config:\n" + pprint.pformat(c.serialize()))
    torch.set_default_device("cuda")
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    torch.cuda.manual_seed_all(c.seed)

    victim_obj_mapping = dict()
    victims_dict = dict()
    for obj in c.objectives:
        mn = obj.victim.cfg.model_name
        if mn not in victim_obj_mapping:
            victim_obj_mapping[mn] = []
        victim_obj_mapping[mn].append(obj)
    for mn, objs in victim_obj_mapping.items():
        victims_dict[mn] = objs[0].victim.setup(
            util.load_model(mn),
            skip_ft_model=not any([o.attack_mult > 0 for o in objs]) \
                or all([isinstance(o, InternalObjectiveOriginalModel) \
                    or isinstance(o, InternalObjectiveCosine) for o in objs]),
        )
    victims = list(victims_dict.values())
    objectives = [o.setup(victims_dict[o.victim.cfg.model_name]) for o in c.objectives]
    tasks = np.unique(sum([o.tasks for o in objectives], []))

    attack_obj_idxs = [i for (i, o) in enumerate(objectives) if o.attack_mult > 0]
    nonattack_obj_idxs = [i for (i, o) in enumerate(objectives) if o.attack_mult == 0]

    n_parts = [victim.count_parts_in_template(o.template, o.victim) for o in objectives]
    assert all([n == n_parts[0] for n in n_parts[1:]])
    n_parts = n_parts[0]

    if c.load_checkpoint is not None:
        load_project, load_run = c.load_checkpoint.split("/")
        result = load_checkpoint(c.s3_bucket, load_project, load_run)
        attack_parts = result["history"][-1]["buffer"]["attacks"][0]["parts"]
        attack = Attack(attack_parts)
    elif c.attack_parts is not None:
        attack = Attack(c.attack_parts)
        if n_parts != len(c.attack_parts):
            raise ValueError("Initial attack has the wrong number of parts.")
        for i in range(n_parts):
            victim.check_and_repair_attack(attack, i, objectives, should_repair=True)
    else:
        if c.attack_part_lens is not None:
            assert n_parts == len(c.attack_part_lens)
        else:
            c.attack_part_lens = [32] * n_parts
        attack = victim.random_attack(objectives, c.attack_part_lens, tasks)
    logger.info("\nINITIAL attack:")
    for i, p in enumerate(attack.parts):
        logger.info(f"PART {i}: {p}")

    if c.iters is None:
        c.iters = max([s.end for s in c.schedule])

    cur_schedule_idx = 0
    schedule = [Settings(c.iters + 1)] if c.schedule is None else c.schedule
    settings = schedule[0]
    for o in objectives:
        o.fluency_mult = settings.fluency_mult
        o.repetition_mult = settings.repetition_mult

    buffer = multi_evaluate(objectives, [attack])

    if torch.isinf(buffer.loss[0]):
        raise ValueError("Initial attack is illegal")

    history = []

    overall_start = time.time()
    checkpoint_again = False

    for it in range(c.iters):
        for s_idx, s in enumerate(schedule):
            if (s_idx > cur_schedule_idx) and (
                schedule[cur_schedule_idx].end <= it < s.end
            ):
                old_fluency_mult = settings.fluency_mult
                old_repetition_mult = settings.repetition_mult
                cur_schedule_idx = s_idx
                settings = s
                if (
                    settings.fluency_mult != old_fluency_mult
                    or settings.repetition_mult != old_repetition_mult
                ):
                    buffer = buffer.subset([0])
                    for v in victims:
                        v.fluency_mult = settings.fluency_mult
                        v.repetition_mult = settings.repetition_mult
                break

        logger.info(f"\nbeginning step {it}")
        start = time.time()

        ordering = [
            attack_obj_idxs[i] for i in torch.randperm(len(attack_obj_idxs))
        ] + nonattack_obj_idxs
        obj_batch = [objectives[i] for i in ordering]

        part_ids = buffer.attacks[0].get_part_ids(obj_batch[0].tokenizer)
        n_tokens = sum([p_ids.shape[0] for p_ids in part_ids])

        if c.token_length_ramp is not None:
            min_tokens = min(
                c.start_tokens
                + (c.min_tokens - c.start_tokens) * it / c.token_length_ramp,
                c.min_tokens,
            )
            max_tokens = min(
                c.start_tokens
                + (c.max_tokens - c.start_tokens) * it / c.token_length_ramp,
                c.max_tokens,
            )
        else:
            min_tokens = c.min_tokens
            max_tokens = c.max_tokens

        update_type_probs = torch.tensor(
            [
                settings.p_delete,
                settings.p_swap,
                settings.p_gcg,
                settings.p_insert,
                settings.p_edge,
            ]
        )
        if n_tokens <= min_tokens:
            # only inserts for too short sequences
            update_type_probs[0] = 0.0
            update_type_probs[1] = 0.0
            update_type_probs[2] = 0.0
        elif n_tokens >= max_tokens:
            # no inserts for too long sequences
            update_type_probs[3] = 0.0
            update_type_probs[4] = 0.0
        update_type_idx = torch.multinomial(update_type_probs, 1).item()
        update_type = ["delete", "swap", "gcg", "insert", "edge_insert"][
            update_type_idx
        ]

        logger.info(update_type)
        if update_type == "gcg":
            candidates = obj_batch[0].gcg_candidates(buffer.attacks[0], settings.k2)
        elif update_type != "delete":
            candidates = obj_batch[0].beast_candidates(buffer.attacks[0], settings.k2)

        if update_type == "delete":
            new_attacks = operators.delete(
                attack=buffer.subset([0]),
                tokenizer=obj_batch[0].tokenizer,
                n=settings.k1,
            )
        elif "insert" in update_type:
            only_edges = False
            if update_type == "edge_insert":
                only_edges = True
            new_attacks = operators.insert(
                attack=buffer.subset([0]),
                candidates=candidates,
                tokenizer=obj_batch[0].tokenizer,
                k1=settings.k1,
                only_edges=only_edges,
            )
        elif "gcg" == update_type or "swap" == update_type:
            new_attacks = operators.swap(
                attack=buffer.subset([0]),
                candidates=candidates,
                tokenizer=obj_batch[0].tokenizer,
                k1=settings.k1,
            )
        else:
            raise ValueError(f"Unknown update type: {update_type}")

        with torch.inference_mode():
            evaluation = multi_evaluate(obj_batch, new_attacks)
        buffer = operators.buffer_retain(
            buffer, evaluation, settings.buffer_size, False
        )
        if torch.isinf(buffer.loss[0]):
            raise ValueError("")

        runtime = time.time() - start
        min_loss_idx = evaluation.loss.argmin()
        log = {
            ("min_loss_" + k): getattr(evaluation, k)[min_loss_idx].item()
            for k in ["xentropy", "repetition", "attack_loss"]
        }
        convergence_reports = [
            x for x in evaluation.convergence_reports if x is not None
        ]
        if len(convergence_reports) > 0:
            log["min_loss_first_prob"] = convergence_reports[0][1].iloc[0]["p_force"]
        log["buffer0_loss"] = buffer.loss[0].item()
        log["runtime"] = runtime

        for i, v in enumerate(victims):
            log[f"n_tokens_victim_{i}"] = sum(
                [
                    len(v.tokenizer.encode(p, add_special_tokens=False))
                    for p in buffer.attacks[0].parts
                ]
            )

        if c.wandb_log and it % c.wandb_log_freq == 0:
            wandb.log(log)

        free, total = torch.cuda.mem_get_info()
        logger.info(f"runtime: {runtime:.3f}")
        logger.info(f"gpu memory usage pct: {100 * (total-free) / total}%")
        logger.info(
            f'attack: {log["min_loss_attack_loss"]:.2f}'
            f' repetition: {log["min_loss_repetition"]:.2f}'
            f' xentropy: {log["min_loss_xentropy"]:.2f}'
            f' perplexity: {np.exp(log["min_loss_xentropy"]):.2f}'
        )
        logger.info(f"buffer.loss[0]: {buffer.loss[0].item():.3f}")
        logger.info(pprint.pformat(evaluation.attacks[min_loss_idx].parts))
        logger.info(
            "prompt: "
            + repr(
                obj_batch[0].template(
                    task="{task}",
                    attack=evaluation.attacks[min_loss_idx],
                    injection="{injection}",
                    victim=obj_batch[0].victim,
                )
            ),
        )

        for report in evaluation.convergence_reports:
            if report is None:
                continue
            info, df = report
            # logger.info(info)
            # logger.info(df[df["flagged"]].drop("flagged", axis=1))

        history.append(
            dict(
                update_type=update_type,
                evaluation=evaluation.serialize(),
                buffer=buffer.serialize(),
            )
        )

        total_runtime = time.time() - overall_start
        out_of_time = c.runtime_limit > 0 and total_runtime > c.runtime_limit
        terminate = (it == c.iters - 1) or out_of_time

        if (
            ((it > 0) and (it % c.checkpoint_freq == 0))
            or terminate
            or checkpoint_again
        ):
            checkpoint_again = False
            best_iter = len(history) - 1
            best_attack = history[best_iter]["buffer"]["attacks"][0]

            generation_ids = []
            generations = []
            prompts = []
            for obj in obj_batch:
                if obj.attack_mult == 0:
                    continue
                for task_data in obj.task_data:
                    generate_out = victim.generate(
                        obj.victim.base_model,
                        obj.tokenizer,
                        [
                            obj.template(
                                Attack(**best_attack),
                                task_data.task,
                                task_data.force_match,
                                obj.victim,
                            )
                        ],
                        max_new_tokens=(
                            c.final_generate_tokens if terminate else c.generate_tokens
                        ),
                    )
                    prompt = generate_out["prompts"][0]
                    prompts.append(prompt)
                    generation_ids.append(generate_out["generation_ids"][0])
                    generations.append(generate_out["generations"][0])
                    logger.info("\n\n")
                    logger.info(f"MODEL: {obj.victim.cfg.short_name}")
                    logger.info(f"TASK: {task_data.task}")
                    logger.info(f"PROMPT: {prompt}")
                    logger.info(f"BEST GENERATION (iter={best_iter}): ")
                    logger.info(generations[-1])

            output = dict(
                best_attack=best_attack,
                generation_ids=generation_ids,
                generations=generations,
                prompts=prompts,
                history=history,
            )
            # output = dict(
            #     config=c.serialize(),
            #     best_attack=best_attack,
            #     generation_ids=generation_ids,
            #     generations=generations,
            #     prompts=prompts,
            #     history=history,
            # )
            output_dir = os.path.join(OUTPUT_DIR, "enhanced_gcg", "output", c.project_name, c.run_name)
            output_path = os.path.join(output_dir, f"iter{it}.pkl")
            logger.info(f"Writing to disk: {output_path}")
            os.makedirs(output_dir, exist_ok=True)
            # try:
            #     with open(output_path, "wb") as f:
            #         pickle.dump(output, f)
            # # sometimes the modal s3 volume thing fails randomly, so we just
            # # try again on the next iteration
            # except PermissionError:
            #     checkpoint_again = True

        if terminate:
            break

    if c.n_postprocess_select > 0:
        logger.info("\nJudging attack success:")
        short_report, long_report = judge.judge(
            output, victims[0], n_select=c.n_postprocess_select
        )
        with open(f"{output_dir}/report_short.json", "w") as f:
            json.dump(short_report, f, indent=4, sort_keys=False)
        with open(f"{output_dir}/report_long.pkl", "wb") as f:
            pickle.dump(long_report, f)

    return output


def is_debug(cfgs):
    names = [c.run_name for c in cfgs]
    if len(np.unique(names)) != len(cfgs):
        raise ValueError("run_name must be unique for each config")

    # if debugging or profiling, run locally, reduce batch size, only attack one task
    try:
        is_debug_run = sys.gettrace() is not None or "profile" in __builtins__
    except TypeError:
        is_debug_run = False

    if is_debug_run:
        warnings.warn("Debug run! wandb_log = False. Running locally.")
        for c in cfgs:
            c.wandb_log = False
    return is_debug_run


@modal_defs.stub.function(**modal_defs.default_params)
def _modal_attack(cfg: AttackConfig):
    os.system("nvidia-smi")
    return attack(cfg)


def modal_attack(cfgs: List[AttackConfig]):
    print(f"Launching {len(cfgs)} jobs")

    use_modal = True
    if is_debug(cfgs):
        use_modal = False

    if use_modal:
        with modal_defs.stub.run():
            return list(_modal_attack.map(cfgs, return_exceptions=True))
    else:
        for c in cfgs:
            return [attack(c) for c in cfgs]


gpu_locks = None


def init_gpu_locks(locks):
    global gpu_locks
    gpu_locks = locks


def _local_attack(cloudpickle_cfg):
    import cloudpickle

    cfg = cloudpickle.loads(cloudpickle_cfg)

    global gpu_locks

    device = None
    for _ in range(50):
        for i in range(len(gpu_locks)):
            if gpu_locks[i].acquire(block=False):
                print("Acquired GPU:", i)
                device = i
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
                break
        if device is None:
            time.sleep(1)
        else:
            break
    if device is None:
        raise SystemError("No GPUs available.")

    try:
        out = attack(cfg)
    finally:
        print("Releasing GPU:", device)
        gpu_locks[device].release()
    return out


def local_attack(cfgs: List[AttackConfig]):
    import cloudpickle

    os.system("nvidia-smi")
    if is_debug(cfgs):
        return [attack(c) for c in cfgs]

    if len(cfgs) == 1:
        return [attack(c) for c in cfgs]

    n_gpus = torch.cuda.device_count()
    gpu_locks = [multiprocessing.Lock() for _ in range(n_gpus)]
    cloudpickled_cfgs = [cloudpickle.dumps(c) for c in cfgs]

    with multiprocessing.Pool(
        n_gpus, initializer=init_gpu_locks, initargs=(gpu_locks,)
    ) as pool:
        return list(pool.map(_local_attack, cloudpickled_cfgs, chunksize=1))
