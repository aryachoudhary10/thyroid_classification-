"""Drive the Kaggle training run from the command line.

    python kaggle_jobs/launch.py push            # start a training session
    python kaggle_jobs/launch.py push --chain    # ...continuing the previous one
    python kaggle_jobs/launch.py watch           # poll until the session ends
    python kaggle_jobs/launch.py fetch           # download log + outputs
    python kaggle_jobs/launch.py run             # push + watch + fetch
    python kaggle_jobs/launch.py status
    python kaggle_jobs/launch.py bundle          # just regenerate kernel_main.py
    python kaggle_jobs/launch.py preflight       # check GPU/internet entitlements

Session chaining
----------------
A Kaggle GPU session is time-capped, so a long run is a chain of sessions.
Version 1 runs fresh; every later push attaches the previous version's output
via ``kernel_sources``, and the kernel copies those checkpoints back into
/kaggle/working before continuing. StageRegistry skips finished folds, so no
GPU work is ever repeated.

Source delivery
---------------
By default the project source is embedded in the kernel script as a base64
tar.gz. Creating a Kaggle *dataset* needs account privileges that pushing a
kernel does not, so embedding removes a whole class of permission and
version-drift problems. ``--dataset-code`` switches to the dataset route.
"""
from __future__ import annotations

import argparse
import base64
import io as _io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_DIR = os.path.join(HERE, "train")

USER = "choudhary15"
CODE_DS = USER + "/dermil-code"
KERNEL = USER + "/dermil-train"
PROBE = USER + "/dermil-gpu-probe"
# Kaggle offers P100 (sm_60) and T4 x2 (sm_75). The preinstalled PyTorch 2.10
# build dropped Pascal support, so a P100 session fails with
# "no kernel image is available for execution on the device". Always ask for T4.
ACCELERATOR = "nvidiaTeslaT4"

THYROID_DS = "safaaqaisi/thyroidxl"
TN5000_DS = "abdullahelafifi/main-data"


# --------------------------------------------------------------------------- #
def kaggle(*args, check=True):
    cmd = [sys.executable, "-m", "kaggle"] + list(args)
    print("$ kaggle " + " ".join(args))
    # Kaggle logs contain non-cp1252 characters; without UTF-8 mode the CLI
    # dies mid-write on Windows and leaves a 0-byte log file.
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip()[:4000])
    if check and r.returncode != 0:
        raise SystemExit("kaggle command failed (%d)" % r.returncode)
    return out


# --------------------------------------------------------------------------- #
BOOTSTRAP = [
    "# -------------------------------------------------------------------",
    "# AUTO-GENERATED -- edit kaggle_jobs/train/train_kernel.py instead.",
    "# The project source is embedded above as a base64 tar.gz and is",
    "# extracted before anything else runs.",
    "# -------------------------------------------------------------------",
    "import base64, io, os, sys, tarfile",
    "_DEST = '/kaggle/working/_code'",
    "os.makedirs(_DEST, exist_ok=True)",
    "_tar = tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PAYLOAD)))",
    "try:",
    "    _tar.extractall(_DEST, filter='data')",
    "except TypeError:",
    "    _tar.extractall(_DEST)",
    "_tar.close()",
    "os.environ['DERMIL_CODE'] = _DEST",
    "sys.path.insert(0, _DEST)",
    "print('embedded source extracted to ' + _DEST)",
    "# -------------------------------------------------------------------",
    "",
]


def build_self_contained_kernel(env=None) -> str:
    """Write kaggle_jobs/train/kernel_main.py with src/ embedded.

    ``env`` is baked in as os.environ defaults, because Kaggle offers no
    way to pass environment variables to a pushed kernel.
    """
    buf = _io.BytesIO()
    n_files = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "src")):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(filenames):
                if fn.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, fn)
                arc = os.path.relpath(full, ROOT).replace(os.sep, "/")
                tar.add(full, arcname=arc)
                n_files += 1
    payload = base64.b64encode(buf.getvalue()).decode("ascii")

    with open(os.path.join(TRAIN_DIR, "train_kernel.py"), encoding="utf-8") as fh:
        body = fh.read()

    lines = ["_PAYLOAD = " + repr(payload)] + BOOTSTRAP
    if env:
        lines.append("import os as _os")
        for k, v in sorted(env.items()):
            lines.append("_os.environ[%r] = %r" % (str(k), str(v)))
        lines.append("print('run parameters: ' + %r)"
                     % str(dict(sorted(env.items()))))
        lines.append("")
    out = os.path.join(TRAIN_DIR, "kernel_main.py")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n" + body)

    kb = os.path.getsize(out) / 1024.0
    print("bundled %d source files -> kernel_main.py (%.0f KB)" % (n_files, kb))
    if kb > 900:
        print("WARNING: generated kernel is large; Kaggle may reject it")
    return out


# --------------------------------------------------------------------------- #
def sync_code(version_notes="update"):
    """Alternative delivery: upload src/ as a Kaggle dataset.

    Requires dataset-creation privileges, which an unverified account does not
    have (the API returns 403). Embedding is the default for that reason.
    """
    d = os.path.join(HERE, "code")
    src_dst = os.path.join(d, "src")
    if os.path.isdir(src_dst):
        shutil.rmtree(src_dst)
    shutil.copytree(os.path.join(ROOT, "src"), src_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(os.path.join(d, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump({"title": "dermil code", "id": CODE_DS,
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)
    out = kaggle("datasets", "status", CODE_DS, check=False)
    exists = "ready" in out.lower()
    if exists:
        kaggle("datasets", "version", "-p", d, "-m", version_notes,
               "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", d, "--dir-mode", "zip")
    print("code dataset synced -> " + CODE_DS)


# --------------------------------------------------------------------------- #
def write_metadata(chain: bool, gpu: bool = True, internet: bool = True,
                   embed: bool = True) -> str:
    sources = [THYROID_DS, TN5000_DS] + ([] if embed else [CODE_DS])
    meta = {
        "id": KERNEL,
        "title": "dermil train",
        "code_file": "kernel_main.py" if embed else "train_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        # torchvision fetches the pretrained ResNet weights at runtime
        "enable_internet": internet,
        "dataset_sources": sources,
        "competition_sources": [],
        # self-reference: attach the previous session's output
        "kernel_sources": [KERNEL] if chain else [],
        "model_sources": [],
        "machine_shape": ACCELERATOR if gpu else None,
    }
    if not gpu:
        meta.pop("machine_shape", None)
    p = os.path.join(TRAIN_DIR, "kernel-metadata.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("metadata: chain=%s gpu=%s (%s) internet=%s embed=%s"
          % (chain, gpu, ACCELERATOR if gpu else "-", internet, embed))
    return p


def push(chain: bool, gpu: bool = True, embed: bool = True, env=None):
    if embed:
        build_self_contained_kernel(env)
    else:
        sync_code()
    write_metadata(chain, gpu=gpu, embed=embed)
    args = ["kernels", "push", "-p", TRAIN_DIR]
    if gpu:
        args += ["--accelerator", ACCELERATOR]
    kaggle(*args)


# --------------------------------------------------------------------------- #
def status() -> str:
    return kaggle("kernels", "status", KERNEL, check=False)


def watch(poll=60, max_hours=13, kernel=None) -> str:
    kernel = kernel or KERNEL
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_hours * 3600:
        s = kaggle("kernels", "status", kernel, check=False).strip()
        s = s.splitlines()[-1] if s else ""
        if s != last:
            print("[%6.1f min] %s" % ((time.time() - t0) / 60, s))
            last = s
        if any(k in s for k in ("COMPLETE", "ERROR", "CANCEL")):
            return s
        time.sleep(poll)
    return "TIMEOUT"


def read_log(out_dir: str) -> str:
    logs = [f for f in os.listdir(out_dir) if f.endswith(".log")]
    if not logs:
        return ""
    raw = open(os.path.join(out_dir, logs[0]), encoding="utf-8",
               errors="ignore").read()
    try:
        text = "\n".join(e.get("data", "") for e in json.loads(raw))
    except Exception:                                          # noqa: BLE001
        text = raw
    with open(os.path.join(out_dir, "clean.log"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def fetch(out_dir=None, tail=70) -> str:
    out_dir = out_dir or os.path.join(TRAIN_DIR, "out")
    os.makedirs(out_dir, exist_ok=True)
    kaggle("kernels", "output", KERNEL, "-p", out_dir, check=False)
    text = read_log(out_dir)
    if text:
        print("\n" + "=" * 74)
        print("TAIL OF SESSION LOG")
        print("=" * 74)
        print("\n".join(text.strip().split("\n")[-tail:]))
    return out_dir


# --------------------------------------------------------------------------- #
def preflight(poll=20):
    """Verify the account really gets a GPU and internet before a long run.

    Kaggle does NOT reject enable_gpu/enable_internet on an unverified account
    -- it silently runs the kernel on CPU with no network, which would waste a
    whole session. This catches that in about two minutes.
    """
    d = os.path.join(HERE, "probe")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump({"id": PROBE, "title": "dermil gpu probe",
                   "code_file": "probe.py", "language": "python",
                   "kernel_type": "script", "is_private": True,
                   "enable_gpu": True, "enable_tpu": False,
                   "enable_internet": True, "dataset_sources": [],
                   "competition_sources": [], "kernel_sources": [],
                   "model_sources": [], "machine_shape": ACCELERATOR},
                  fh, indent=2)
    kaggle("kernels", "push", "-p", d, "--accelerator", ACCELERATOR)
    final = watch(poll=poll, max_hours=1, kernel=PROBE)
    print("probe finished:", final)

    out = os.path.join(d, "out")
    os.makedirs(out, exist_ok=True)
    kaggle("kernels", "output", PROBE, "-p", out, check=False)
    text = read_log(out)
    gpu_ok = "cuda available: True" in text
    net_ok = "reachable: True" in text
    wts_ok = "loaded OK" in text
    print("\n" + "=" * 74)
    print("  GPU available       : %s" % ("YES" if gpu_ok else "NO"))
    print("  internet available  : %s" % ("YES" if net_ok else "NO"))
    print("  pretrained weights  : %s" % ("YES" if wts_ok else "NO"))
    print("=" * 74)
    if not (gpu_ok and net_ok):
        print("\nThis account cannot run the training kernel yet.")
        print("Verify your phone number at https://www.kaggle.com/settings")
        print("(GPU, internet and dataset creation are all gated behind it).")
    return gpu_ok and net_ok


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "watch", "fetch", "run", "status",
                                       "bundle", "sync", "meta", "preflight"])
    ap.add_argument("--chain", action="store_true",
                    help="attach the previous session output and continue")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dataset-code", action="store_true",
                    help="deliver source via the dermil-code dataset")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--env", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="run parameter baked into the kernel (repeatable)")
    a = ap.parse_args()
    embed = not a.dataset_code
    env = dict(kv.split("=", 1) for kv in a.env) if a.env else None

    if a.action == "bundle":
        build_self_contained_kernel(env)
    elif a.action == "sync":
        sync_code()
    elif a.action == "meta":
        if embed:
            build_self_contained_kernel()
        write_metadata(a.chain, gpu=not a.no_gpu, embed=embed)
    elif a.action == "preflight":
        preflight()
    elif a.action == "push":
        push(a.chain, gpu=not a.no_gpu, embed=embed, env=env)
    elif a.action == "watch":
        print("final:", watch(a.poll))
    elif a.action == "fetch":
        fetch()
    elif a.action == "status":
        status()
    elif a.action == "run":
        if not a.skip_preflight and not a.no_gpu:
            if not preflight():
                raise SystemExit("preflight failed -- not starting the run")
        push(a.chain, gpu=not a.no_gpu, embed=embed, env=env)
        print("final:", watch(a.poll))
        fetch()


if __name__ == "__main__":
    main()
