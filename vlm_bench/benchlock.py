import json, platform, subprocess, os
from pathlib import Path

def write_bench_lock(path: Path):
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES",""),
        "git": _git_rev(),
        "pip_freeze": _pip_freeze()
    }
    path.write_text(json.dumps(info, indent=2))

def _pip_freeze():
    try:
        return subprocess.check_output(["python","-m","pip","freeze"], text=True)
    except Exception:
        return ""
def _git_rev():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception:
        return "unknown"
