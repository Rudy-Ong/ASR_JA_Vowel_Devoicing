"""
scripts/sweep_runner.py
────────────────────────
Per-GPU sequential runner + hang watchdog for the phone3 BS×LR×DW sweep.

Each invocation owns one GPU and works through its assigned list of
(batch_size, lr, devo_weight) configs in order, launching train_r3.py as a
subprocess with the right env vars. While a run is active, it polls that
run's _train.log mtime every --check-interval seconds; if it goes stale for
longer than --hang-timeout, the subprocess is killed and relaunched — with
RESUME_FROM pointed at the best checkpoint saved so far, if one exists,
otherwise from scratch — up to --max-retries times before giving up on that
config and moving to the next one.

This *is* the hang checker: no separate coordinator process is needed, one
instance per GPU is fully self-contained.

Usage:
    python scripts/sweep_runner.py --gpu 0 \\
        --configs '8,0.001,1' '8,0.001,5' '16,0.001,1' '16,0.001,5'
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_PATH_RE = re.compile(r'(?:Streaming epoch logs to tmux pane\s+→\s+|Logs written to\s+)(\S+)')
TRAIN_START_RE = re.compile(r'=== Training start: (\S+)')


def parse_config(s: str) -> tuple[int, float, float]:
    bs, lr, dw = s.split(',')
    return int(bs), float(lr), float(dw)


def run_one(gpu: int, bs: int, lr: float, dw: float, resume_from: str | None,
            check_interval: int, hang_timeout: int, enc_padding_mask: bool,
            cpu_cores: str | None, torch_threads: int) -> tuple[bool, str | None]:
    """Launch one train_r3.py subprocess, watch it, kill on hang.

    Returns (completed_cleanly, checkpoint_path_or_None) so the caller can
    decide whether/how to retry.
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['SWEEP_BATCH_SIZE'] = str(bs)
    env['SWEEP_LR'] = str(lr)
    env['SWEEP_DEVO_WEIGHT'] = str(dw)
    env['SWEEP_NAME_SUFFIX'] = f'dw{dw:g}_stratified'
    env['TORCH_NUM_THREADS'] = str(torch_threads)
    env['OMP_NUM_THREADS'] = str(torch_threads)
    env['MKL_NUM_THREADS'] = str(torch_threads)
    if enc_padding_mask:
        env['ENC_PADDING_MASK'] = '1'
    if resume_from:
        env['RESUME_FROM'] = resume_from
    else:
        env.pop('RESUME_FROM', None)

    tag = f'[gpu{gpu} bs{bs} lr{lr} dw{dw:g}]'
    print(f'{tag} launching{" (RESUME_FROM=" + resume_from + ")" if resume_from else ""}', flush=True)

    cmd = [sys.executable, 'train_r3.py']
    if cpu_cores:
        cmd = ['taskset', '-c', cpu_cores] + cmd

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    state = {'log_path': None, 'model_name': None, 'last_mtime': time.time()}
    state_lock = threading.Lock()

    def reader() -> None:
        for line in proc.stdout:
            print(f'{tag} {line.rstrip()}', flush=True)
            m = LOG_PATH_RE.search(line)
            if m:
                with state_lock:
                    if state['log_path'] is None:
                        state['log_path'] = Path(m.group(1))
                        state['last_mtime'] = time.time()
            m2 = TRAIN_START_RE.search(line)
            if m2:
                with state_lock:
                    state['model_name'] = m2.group(1)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    def kill_proc() -> None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    while proc.poll() is None:
        time.sleep(min(check_interval, 5))
        now = time.time()
        with state_lock:
            log_path = state['log_path']
            last_mtime = state['last_mtime']
            model_name = state['model_name']

        if log_path is not None and log_path.exists():
            mtime = log_path.stat().st_mtime
            if mtime > last_mtime:
                with state_lock:
                    state['last_mtime'] = mtime
                continue
            if now - last_mtime > hang_timeout:
                print(f'{tag} HANG DETECTED: {log_path.name} stale for '
                      f'{int(now - last_mtime)}s (> {hang_timeout}s). Killing pid={proc.pid}.',
                      flush=True)
                kill_proc()
                reader_thread.join(timeout=5)
                return False, _model_name_to_ckpt(model_name)
        elif now - last_mtime > hang_timeout:
            print(f'{tag} HANG DETECTED: no log file after {hang_timeout}s. '
                  f'Killing pid={proc.pid}.', flush=True)
            kill_proc()
            reader_thread.join(timeout=5)
            return False, None

    reader_thread.join(timeout=10)
    rc = proc.returncode
    ckpt = _model_name_to_ckpt(state['model_name'])
    if rc == 0:
        print(f'{tag} completed cleanly (model={state["model_name"]})', flush=True)
        return True, ckpt
    print(f'{tag} exited with code {rc} (model={state["model_name"]})', flush=True)
    return False, ckpt


def _model_name_to_ckpt(model_name: str | None) -> str | None:
    if not model_name:
        return None
    ckpt = PROJECT_ROOT / 'models' / f'{model_name}.pt'
    return str(ckpt) if ckpt.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--configs', nargs='+', required=True,
                     help='List of "bs,lr,dw" strings, run in order on this GPU.')
    ap.add_argument('--check-interval', type=int, default=300,
                     help='Seconds between hang checks (default 300).')
    ap.add_argument('--hang-timeout', type=int, default=900,
                     help='Seconds a _train.log can go stale before considered hung (default 900).')
    ap.add_argument('--max-retries', type=int, default=5,
                     help='Max relaunches per config before giving up on it (default 5).')
    ap.add_argument('--enc-padding-mask', action='store_true',
                     help='Set ENC_PADDING_MASK=1 for every run (decoder cross-attention padding fix).')
    ap.add_argument('--cpu-cores', type=str, default=None,
                     help='taskset core range for this GPU\'s subprocess, e.g. "0-11". '
                          'Overrides the auto-derived range from --num-gpus.')
    ap.add_argument('--num-gpus', type=int, default=4,
                     help='Total concurrent sweep_runner instances sharing this box\'s '
                          'cores. Used to auto-derive a disjoint --cpu-cores range per '
                          '--gpu (e.g. 48 cores / 4 gpus -> 12 cores each) when '
                          '--cpu-cores is not given explicitly (default 4).')
    ap.add_argument('--torch-threads', type=int, default=6,
                     help='Cap on TORCH/OMP/MKL_NUM_THREADS passed to train_r3.py, to avoid '
                          'per-process thread oversubscription on a shared box (default 6).')
    args = ap.parse_args()

    cpu_cores = args.cpu_cores
    if cpu_cores is None and args.num_gpus > 0:
        total_cores = os.cpu_count() or 1
        per_gpu = max(1, total_cores // args.num_gpus)
        start = args.gpu * per_gpu
        end = min(start + per_gpu, total_cores) - 1
        if end >= start:
            cpu_cores = f'{start}-{end}'
            print(f'[gpu{args.gpu}] auto-pinning to cores {cpu_cores} '
                  f'(--cpu-cores not given, derived from --num-gpus={args.num_gpus})', flush=True)

    configs = [parse_config(c) for c in args.configs]
    print(f'[gpu{args.gpu}] {len(configs)} configs queued: {configs}', flush=True)

    for bs, lr, dw in configs:
        resume_from = None
        for attempt in range(1, args.max_retries + 1):
            ok, ckpt = run_one(
                args.gpu, bs, lr, dw, resume_from,
                args.check_interval, args.hang_timeout, args.enc_padding_mask,
                cpu_cores, args.torch_threads,
            )
            if ok:
                break
            resume_from = ckpt
            print(f'[gpu{args.gpu} bs{bs} lr{lr} dw{dw:g}] retry {attempt}/{args.max_retries} '
                  f'{"with RESUME_FROM=" + ckpt if ckpt else "from scratch"}', flush=True)
        else:
            print(f'[gpu{args.gpu} bs{bs} lr{lr} dw{dw:g}] giving up after '
                  f'{args.max_retries} attempts, moving on.', flush=True)

    print(f'[gpu{args.gpu}] all configs done.', flush=True)


if __name__ == '__main__':
    main()
