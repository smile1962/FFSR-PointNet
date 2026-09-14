import os
import subprocess
import sys


PYTHON = r"E:\Anaconda\envs\pytorch\python.exe"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script, cwd, defaults=None, extra=None):
    cmd = [PYTHON, "-u", script]
    if defaults:
        cmd.extend(defaults)
    if extra:
        cmd.extend(extra)
    env = os.environ.copy()
    print("Running:", " ".join(cmd), f"in {cwd}")
    return subprocess.call(cmd, cwd=cwd, env=env)

