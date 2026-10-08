"""Resumable driver for the unmodified official QD-DETR baseline at fork rev cbbb5be
(legacy/task6-qd-detr-cbbb5be: upstream fixes bf0e88f T2V attention mask, 35fcbaf CASTELLA timestamps).
Copied from results/qd-clotho-pretrain-20261007 (rev 481d60b run); changes: data-file hashes in the
stage identity and DONE records, and the `run` command (CASTELLA-only seeds, then the Clotho-Moment pretraining -> CASTELLA fine-tuning chain).

Runs inside the dcase26-task6:py271-cu128 container with the working directory set to the
baseline root (relative data/feature paths in the configs resolve there).  The baseline's own
src/ is imported, not edited.  The training loop below is the baseline's train.main() + train()
with three additions only:
  * after every epoch the full state (model, optimizer, scheduler, next epoch, best score,
    python/numpy/torch CPU/CUDA RNG, train.log/val.log sizes) is saved atomically to
    resume_state.pt; rerunning the same command continues from it,
  * best_checkpoint.pth is written atomically (same dict as basic_utils.save_checkpoint),
  * a DONE file marks a finished stage, which is then skipped.
The state records the config, init-checkpoint, code and data hashes; a mismatch refuses to resume.

  python qd_resumable.py run --run-dir <dir> --castella-only-seeds 2023 ... --pretrain-finetune-seeds 2023 ...
      first per CASTELLA-only seed: CASTELLA only training -> val and test inference of its best checkpoint,
      then the chain below for the pretrain -> finetune seeds.
  python qd_resumable.py chain --run-dir <dir> --seeds 2023 2024 ...
      per seed: Clotho-Moment pretraining -> CASTELLA fine-tuning (initialized from the pretraining best
      checkpoint the way the baseline's --resume does) -> val and test inference of the fine-tuning best
      checkpoint.  Everything runs in one process so the GPU stays occupied between stages.
  A finished inference (DONE) is never run again, so the test split is scored once per seed.

QD_STOP_AFTER_EPOCH=<n> (test only) exits with code 75 right after epoch n's state is saved.
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import pprint
import random
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

BASELINE = Path.cwd()
sys.path.insert(0, str(BASELINE / "src"))

import train as base_train  # noqa: E402  (baseline src/train.py)
from basic_utils import rename_latest_to_best, write_log  # noqa: E402
from config import BaseOptions  # noqa: E402
from dataset import StartEndDataset, start_end_collate  # noqa: E402
from evaluate import eval_epoch, setup_model, start_inference  # noqa: E402
from model_utils import count_parameters  # noqa: E402

logger = logging.getLogger("qd_resumable")
EXIT_TEST_STOP = 75


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha():
    h = hashlib.sha256()
    for p in [Path(__file__).resolve()] + sorted((BASELINE / "src").glob("*.py")):
        h.update(p.name.encode())
        h.update(sha256(p).encode())
    return h.hexdigest()


def atomic_torch_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_opt(cfg):
    manager = BaseOptions(str(cfg))
    manager.parse()
    return manager.option


def file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def truncate(path, size):
    if os.path.exists(path):
        with open(path, "r+b") as f:
            f.truncate(size)


def train_stage(cfg, init=None):
    """Baseline main()+train() for one config, resumable.  Returns the DONE record."""
    opt = load_opt(cfg)
    out = Path(opt.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "DONE.json"
    if done.exists():
        logger.info(f"{out.name}: DONE, skip")
        return json.loads(done.read_text())
    if opt.model_ema:
        raise SystemExit("model_ema is not supported by this driver")

    identity = {
        "config_sha256": sha256(cfg),
        "init_checkpoint_sha256": sha256(init) if init else None,
        "code_sha256": code_sha(),
        "data_sha256": {"train": sha256(opt.train_path), "val": sha256(opt.val_path)},
    }
    state_path = out / "resume_state.pt"
    state = None
    if state_path.exists():
        state = torch.load(state_path, weights_only=False)
        if state["identity"] != identity:
            raise SystemExit(f"{out.name}: resume identity mismatch\nstate {state['identity']}\nnow   {identity}")

    # --- baseline main(): seed, datasets, model (identical order, so RNG use matches) ---
    logger.info("Setup config, data and model...")
    base_train.set_seed(opt.seed)
    dataset_config = EasyDict(
        data_path=opt.train_path,
        ctx_mode=opt.ctx_mode,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        a_feat_type=opt.a_feat_type,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    )
    train_dataset = StartEndDataset(**dataset_config)
    copied_eval_config = copy.deepcopy(dataset_config)
    copied_eval_config.data_path = opt.val_path
    val_dataset = StartEndDataset(**copied_eval_config)
    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    count_parameters(model, verbose=True)

    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        start_epoch = state["next_epoch"]
        prev_best_score = state["prev_best_score"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        truncate(opt.train_log_filepath, state["log_sizes"]["train"])
        truncate(opt.eval_log_filepath, state["log_sizes"]["val"])
        logger.info(f"{out.name}: resumed at epoch {start_epoch + 1} (best R1@0.7 {prev_best_score})")
    else:
        if init is not None:
            checkpoint = torch.load(init, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            logger.info("Loaded model checkpoint: {}".format(init))
        start_epoch, prev_best_score = 0, 0
        truncate(opt.train_log_filepath, 0)  # leftovers of an attempt killed inside epoch 1
        truncate(opt.eval_log_filepath, 0)

    # --- baseline train() ---
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    stop_after = int(os.environ.get("QD_STOP_AFTER_EPOCH", "0"))
    for epoch_i in range(start_epoch, opt.n_epoch):
        base_train.train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            with torch.no_grad():
                metrics, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, criterion)
            write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode="val")
            logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
            stop_score = metrics["brief"]["MR-full-R1@0.7"]
            if stop_score > prev_best_score:
                prev_best_score = stop_score
                atomic_torch_save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }, opt.ckpt_filepath)
                logger.info("The checkpoint file has been updated.")
                rename_latest_to_best(latest_file_paths)

        atomic_torch_save({
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "next_epoch": epoch_i + 1,
            "prev_best_score": prev_best_score,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "log_sizes": {"train": file_size(opt.train_log_filepath), "val": file_size(opt.eval_log_filepath)},
        }, state_path)
        if stop_after and epoch_i + 1 == stop_after:
            logger.info(f"QD_STOP_AFTER_EPOCH={stop_after}: simulated interruption")
            raise SystemExit(EXIT_TEST_STOP)

    if not os.path.exists(opt.ckpt_filepath):  # the baseline also saves nothing if R1@0.7 never exceeds 0
        raise SystemExit(f"{out.name}: no epoch had val R1@0.7 > 0, so there is no best checkpoint")
    best = torch.load(opt.ckpt_filepath, weights_only=False)
    metrics_file = out / f"best_{opt.dset_name}_val_preds_metrics.json"
    record = {
        **identity,
        "best_epoch": best["epoch"] + 1,
        "best_checkpoint_sha256": sha256(opt.ckpt_filepath),
        "best_val_brief": json.loads(metrics_file.read_text())["brief"],
        "init": str(init) if init else None,
    }
    atomic_write_text(done, json.dumps(record, indent=1) + "\n")
    logger.info(f"{out.name}: DONE {record['best_val_brief']}")
    return record


def eval_stage(cfg, ckpt, split, out):
    """Baseline evaluate.py --split <split> into its own folder; never repeated once DONE."""
    out = Path(out)
    done = out / "DONE.json"
    if done.exists():
        logger.info(f"{out.name}: DONE, skip (not rescored)")
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    opt = load_opt(cfg)
    opt.results_dir = str(out)
    opt.model_path = str(ckpt)
    opt.eval_split_name = split
    start_inference(opt)
    record = {
        "split": split,
        "checkpoint_sha256": sha256(ckpt),
        "config_sha256": sha256(cfg),
        "data_sha256": sha256(opt.val_path if split == "val" else opt.test_path),
        "brief": json.loads((out / "submission_metrics.json").read_text())["brief"],
    }
    atomic_write_text(done, json.dumps(record, indent=1) + "\n")
    logger.info(f"{out.name}: {record['brief']}")
    return record


def chain(run_dir, seeds):
    run_dir = Path(run_dir)
    for seed in seeds:
        p_cfg = run_dir / "configs" / f"clotho_pretrain_s{seed}.yml"
        f_cfg = run_dir / "configs" / f"castella_finetune_s{seed}.yml"
        p = train_stage(p_cfg)
        p_ckpt = run_dir / "results" / f"clotho_pretrain_s{seed}" / "best_checkpoint.pth"
        if sha256(p_ckpt) != p["best_checkpoint_sha256"]:
            raise SystemExit(f"clotho_pretrain_s{seed} best checkpoint hash differs from its DONE record")
        train_stage(f_cfg, init=p_ckpt)
        f_ckpt = run_dir / "results" / f"castella_finetune_s{seed}" / "best_checkpoint.pth"
        eval_stage(f_cfg, f_ckpt, "val", run_dir / "results" / f"castella_finetune_s{seed}" / "eval_val")
        eval_stage(f_cfg, f_ckpt, "test", run_dir / "results" / f"castella_finetune_s{seed}" / "eval_test")
        logger.info(f"seed {seed}: complete")
    logger.info("chain complete")


def castella_only(run_dir, seeds):
    run_dir = Path(run_dir)
    for seed in seeds:
        a_cfg = run_dir / "configs" / f"castella_only_s{seed}.yml"
        train_stage(a_cfg)
        a_ckpt = run_dir / "results" / f"castella_only_s{seed}" / "best_checkpoint.pth"
        eval_stage(a_cfg, a_ckpt, "val", run_dir / "results" / f"castella_only_s{seed}" / "eval_val")
        eval_stage(a_cfg, a_ckpt, "test", run_dir / "results" / f"castella_only_s{seed}" / "eval_test")
        logger.info(f"castella_only seed {seed}: complete")
    logger.info("CASTELLA-only seeds complete")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--castella-only-seeds", type=int, nargs="*", default=[])
    r.add_argument("--pretrain-finetune-seeds", type=int, nargs="*", default=[])
    c = sub.add_parser("chain")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--seeds", type=int, nargs="+", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--init")
    args = ap.parse_args()
    if args.cmd == "run":
        castella_only(args.run_dir, args.castella_only_seeds)
        chain(args.run_dir, args.pretrain_finetune_seeds)
    elif args.cmd == "chain":
        chain(args.run_dir, args.seeds)
    else:
        train_stage(Path(args.config), args.init)


if __name__ == "__main__":
    main()
