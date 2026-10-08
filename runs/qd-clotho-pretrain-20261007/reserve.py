#!/usr/bin/env python3
"""GPU0 reservation for the QD-DETR Clotho-Moment pretrain -> CASTELLA finetune chain (2026-10-07).

Waits until GPU0 has a free slot, then starts one container that runs
`qd_resumable.py chain` for all seeds (P -> F -> val -> test per seed, resumable).

Slot rule (GPU0 runs at most SLOTS jobs; small-kernel jobs only time-share beyond that):
start when our CV5 fold containers on GPU0 (t0004-cv5*) number <= SLOTS - 1, no other process
uses >= 1 GiB there, and >= 4 GiB is free, on CONFIRM consecutive checks POLL s apart.
Once this chain runs, the CV5 queue sees it as foreign work and adds no new GPU0 folds until
it ends; folds already running continue.

Resumable: rerun the same command.  A running chain container is left alone; an interrupted one
(exit 137/143, reboot, docker stop) is started again when a slot is free and resumes from the
per-stage resume_state.pt; finished stages are skipped.  Any other non-zero exit stops the
waiter for inspection.  A HOLD file in the run directory stops new launches only.

  setsid --fork python3 reserve.py >> logs/reserve.log 2>&1 < /dev/null
"""
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parent
BASELINE = "/home/soyeong/dcase2026/legacy/task6-qd-detr"
CLOTHO_FEAT = "/home/soyeong/dcase2026/A_Kim_v1(0923)/server/downloads/clotho-msclap-zenodo-17129257/extracted/features/clotho-moment"
GPU0 = "GPU-a05bb29b-3af6-f296-441d-fff3dffa21bc"
IMAGE = "dcase26-task6:py271-cu128"
NAME = "qd-clotho-pt-chain"
SEEDS = [2023, 2024, 2025, 2026, 2027]
CV5_PREFIX = "t0004-cv5"
SLOTS = 2
MIN_FREE_MIB = 4096
FOREIGN_MAX_MIB = 1024
POLL = 10
CONFIRM = 3
# 2026-10-07 06:10Z user: QD-DETR before the CV5 folds on GPU0.  The CV5 queue's GPU0 slots were
# lowered to 1 so it does not refill the first slot that frees; restored to 2 when this chain ends.
CV5_GPUS = Path("/home/soyeong/dcase2026/results/t0004-cv5/queue/gpus.json")
CV5_GPUS_AFTER = {GPU0: 2}


def log(msg):
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}", flush=True)


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def complete():
    return all((RUN / "results" / f"F_s{s}" / "eval_test" / "DONE.json").exists() for s in SEEDS)


def restore_cv5_slots():
    if json.loads(CV5_GPUS.read_text()) == {GPU0: 1}:
        tmp = CV5_GPUS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(CV5_GPUS_AFTER) + "\n")
        tmp.replace(CV5_GPUS)
        log(f"restored CV5 GPU0 slots: {CV5_GPUS_AFTER}")


def ours():
    """(running, exit code) of the chain container, or None."""
    r = sh("docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", NAME)
    if r.returncode != 0:
        return None
    running, code = r.stdout.split()
    return running == "true", int(code)


def cv5_on_gpu0():
    n = 0
    for name in sh("docker", "ps", "--format", "{{.Names}}").stdout.split():
        if not name.startswith(CV5_PREFIX):
            continue
        dev = sh("docker", "inspect", "-f", "{{json .HostConfig.DeviceRequests}}", name).stdout
        ids = [i for req in (json.loads(dev) if dev.strip() else None) or [] for i in (req.get("DeviceIDs") or [])]
        n += GPU0 in ids
    return n


def foreign_mib():
    """GPU0 memory of processes outside the CV5 containers."""
    ids = sh("docker", "ps", "-q", "--no-trunc", "--filter", f"name=^{CV5_PREFIX}").stdout.split()
    r = sh("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits")
    total = 0
    for line in r.stdout.splitlines():
        uuid, pid, mib = [x.strip() for x in line.split(",")]
        if uuid != GPU0:
            continue
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError:
            cgroup = ""
        if not any(i in cgroup for i in ids):
            total += int(mib)
    return total


def free_mib():
    r = sh("nvidia-smi", "-i", GPU0, "--query-gpu=memory.free", "--format=csv,noheader,nounits")
    return int(r.stdout.strip()) if r.returncode == 0 else 0


def slot_free():
    cv5, foreign, free = cv5_on_gpu0(), foreign_mib(), free_mib()
    return cv5 <= SLOTS - 1 and foreign < FOREIGN_MAX_MIB and free >= MIN_FREE_MIB, (cv5, foreign, free)


def launch():
    sh("docker", "rm", "-f", NAME)
    seeds = " ".join(map(str, SEEDS))
    inner = (f"python {RUN}/qd_resumable.py chain --run-dir {RUN} --seeds {seeds} >> {RUN}/logs/chain.log 2>&1; "
             f"rc=$?; echo \"EXIT_STATUS=$rc $(date -u +%FT%TZ)\" >> {RUN}/logs/chain.log; exit $rc")
    r = sh("docker", "run", "-d", "--name", NAME, "--gpus", f"device={GPU0}", "--user", "1002:1002",
           "--cpus", "4", "--shm-size", "8g", "--network", "none", "-e", "HOME=/tmp", "-e", "TQDM_MININTERVAL=60",
           "-v", f"{CLOTHO_FEAT}:/feat/clotho:ro", "-v", f"{BASELINE}:{BASELINE}:ro", "-v", f"{RUN}:{RUN}:rw",
           "-w", BASELINE, "--entrypoint", "", IMAGE, "bash", "-c", inner)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def main():
    (RUN / "logs").mkdir(exist_ok=True)
    lock = open(RUN / "logs" / "reserve.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another reserve.py holds the lock")
    log(f"reserve start (seeds {SEEDS}, GPU0 slots {SLOTS})")
    streak, last = 0, None
    while True:
        if complete():
            log("chain complete; reserve exits")
            restore_cv5_slots()
            return
        state = ours()
        if state and state[0]:
            streak = 0
            time.sleep(POLL * 6)
            continue
        if state and not state[0]:
            code = state[1]
            if code == 0 and complete():
                continue
            if code not in (0, 137, 143):
                log(f"chain container exited {code}; stopping for inspection (see logs/chain.log)")
                sys.exit(1)
            if last != ("exited", code):
                log(f"chain container exited {code}; will relaunch to resume when a slot is free")
                last = ("exited", code)
        if (RUN / "HOLD").exists():
            if last != "hold":
                log("HOLD present; not launching")
                last = "hold"
            streak = 0
            time.sleep(POLL)
            continue
        ok, info = slot_free()
        streak = streak + 1 if ok else 0
        if not ok and last != ("wait", info[0], info[1] >= FOREIGN_MAX_MIB):
            log(f"waiting: cv5 on GPU0 {info[0]}/{SLOTS}, foreign {info[1]} MiB, free {info[2]} MiB")
            last = ("wait", info[0], info[1] >= FOREIGN_MAX_MIB)
        if streak >= CONFIRM:
            ok, msg = launch()
            log(f"{'launched' if ok else 'LAUNCH FAILED'} {NAME} on GPU0 (cv5 {info[0]}, free {info[2]} MiB): {msg[:80]}")
            if not ok:
                sys.exit(1)
            streak, last = 0, None
            time.sleep(POLL * 6)
            continue
        time.sleep(POLL)


if __name__ == "__main__":
    main()
