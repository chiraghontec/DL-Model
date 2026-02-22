#!/usr/bin/env python3
"""Watch training progress and notify when target epoch completes.

Usage: training_watch.py --target 30 --interval 30

Behavior:
- Polls `checkpoints/` for highest epoch saved (files containing `epoch_<n>`).
- Falls back to scanning `train_resume_11_to_30.log` for an `Epoch X/Y` header
  and the matching Train/Val summary.
- When the target epoch is detected as completed, sends a macOS notification
  using `osascript` and exits. If the training process died early, it notifies
  that the run stopped prematurely.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def max_epoch_from_checkpoints(cp_dir: Path):
    if not cp_dir.exists():
        return 0
    best = 0
    for p in cp_dir.iterdir():
        if p.suffix != '.pth':
            continue
        m = re.search(r'epoch_(\d+)', p.name)
        if m:
            n = int(m.group(1))
            if n > best:
                best = n
    return best


def epoch_in_log(log_path: Path, target: int):
    if not log_path.exists():
        return False, ''
    text = log_path.read_text(errors='ignore')
    # look for header "Epoch 30/30" or "Epoch 30/"
    header_re = re.compile(rf'^Epoch\s*{target}/', re.M)
    if header_re.search(text):
        # try to find the last Train/Val summary after that header
        # find position of header
        pos = max(m.start() for m in header_re.finditer(text))
        tail = text[pos:]
        summary = re.search(r'(Train loss:.*|Val loss:.*)', tail)
        return True, summary.group(0) if summary else ''
    return False, ''


def notify_mac(title: str, message: str):
    try:
        cmd = ['osascript', '-e', f'display notification "{message}" with title "{title}"']
        subprocess.run(cmd, check=False)
    except Exception:
        # best-effort; ignore errors
        pass


def is_pid_alive(pid_file: Path):
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', type=int, default=30, help='Target epoch to wait for')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Poll interval in seconds')
    parser.add_argument('--checkpoints', '-c', default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--log', '-l', default='train_resume_11_to_30.log', help='Training log file to scan')
    parser.add_argument('--pidfile', '-p', default='train_resume.pid', help='Training run PID file')
    parser.add_argument('--watch-pidfile', default='training_watch.pid', help='This watcher PID file')
    args = parser.parse_args()

    cp_dir = Path(args.checkpoints)
    log_path = Path(args.log)
    pid_file = Path(args.pidfile)
    watch_pid = Path(args.watch_pidfile)

    # write our pid
    try:
        watch_pid.write_text(str(os.getpid()))
    except Exception:
        pass

    target = args.target
    interval = args.interval

    print(f"Monitoring for epoch {target} (checkpoints={cp_dir}, log={log_path})")
    sys.stdout.flush()

    while True:
        best = max_epoch_from_checkpoints(cp_dir)
        if best >= target:
            msg = f'Training finished: reached epoch {best} (target {target}).'
            print(msg)
            notify_mac('Training Complete', msg)
            try:
                watch_pid.unlink()
            except Exception:
                pass
            return

        # check log for header/summary
        found, summary = epoch_in_log(log_path, target)
        if found:
            msg = f'Training finished (log): epoch {target} detected. {summary}'
            print(msg)
            notify_mac('Training Complete', msg)
            try:
                watch_pid.unlink()
            except Exception:
                pass
            return

        # if training process pidfile exists, check process aliveness
        alive = is_pid_alive(pid_file)
        if alive is False:
            # process not alive; report stopped status and current best epoch
            msg = f'Training process stopped prematurely. Last saved epoch: {best}.'
            print(msg)
            notify_mac('Training Stopped', msg)
            try:
                watch_pid.unlink()
            except Exception:
                pass
            return

        # otherwise sleep and retry
        time.sleep(interval)


if __name__ == '__main__':
    main()
