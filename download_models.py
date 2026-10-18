# Download the models this workflow needs, from ModelScope (HuggingFace is unreachable here).
#
#   python download_models.py             # download everything (parallel, resumable)
#   python download_models.py --check     # only report what is missing
#   python download_models.py --conns 16  # more parallel connections
#
# Files land directly in the ComfyUI model folders the workflow expects.
# Interrupted downloads resume: finished chunks are recorded in a .parts.json sidecar,
# and each chunk is written straight into its final offset in the destination file.

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

COMFY_ROOT = r"D:\Al\ComfyUI-aki-v3.2\ComfyUI"
API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
UA = "Mozilla/5.0"
DEFAULT_CONNS = 8
DEFAULT_CHUNK = 64 * 1024 * 1024
RETRIES = 6
READ = 4 * 1024 * 1024

# (repo, file path inside the repo, destination folder, destination file name)
#
# The text encoder is the GGUF build rather than the 8.74 GB fp8 safetensors one:
# this machine has 15.8 GB RAM, so a ~17 GB peak (fp8 encoder + unet) cannot fit,
# while the GGUF encoder keeps the peak near 13 GB. The mmproj file is renamed so
# ComfyUI-GGUF's fuzzy matcher pairs it with the encoder (it matches on the encoder
# file name minus the quant suffix).
MODELS = [
    (
        "QuantStack/Qwen-Image-Edit-2509-GGUF",
        "Qwen-Image-Edit-2509-Q3_K_S.gguf",
        "diffusion_models",
        "Qwen-Image-Edit-2509-Q3_K_S.gguf",
    ),
    (
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "text_encoders",
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    ),
    (
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "mmproj-BF16.gguf",
        "text_encoders",
        "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
        "vae",
        "qwen_image_vae.safetensors",
    ),
    (
        "lightx2v/Qwen-Image-Lightning",
        "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        "loras",
        "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
    ),
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def url_for(repo: str, path: str) -> str:
    return API.format(repo=repo, path=urllib.parse.quote(path))


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
    return int(cr.split("/")[-1]) if "/" in cr else 0


class Progress:
    """Thread-safe byte counter with a periodic speed/ETA line."""

    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.done = 0
        self.lock = threading.Lock()
        self.start = time.time()
        self.last = 0.0
        self.last_done = 0

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last > 3:
                inst = (self.done - self.last_done) / max(now - self.last, 1e-6)
                self.last, self.last_done = now, self.done
                pct = f"{100 * self.done / self.total:5.1f}%" if self.total else "  ?  "
                eta = (self.total - self.done) / inst if inst else 0
                print(
                    f"\r  {self.label} {pct} {human(self.done)}/{human(self.total)} "
                    f"{human(inst)}/s eta {eta / 60:4.1f} min",
                    end="",
                    flush=True,
                )

    def finish(self) -> None:
        el = time.time() - self.start
        print(f"\r  {self.label} 100.0% {human(self.done)}/{human(self.total)} "
              f"avg {human(self.done / max(el, 1e-6))}/s in {el / 60:.1f} min")


def fetch_chunk(url: str, dest: str, start: int, end: int, prog: Progress) -> bool:
    """Write bytes [start, end) of `url` into `dest` at that offset."""
    span = end - start
    for attempt in range(RETRIES):
        got = 0
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end - 1}"}
            )
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "r+b") as fh:
                fh.seek(start)
                while got < span:
                    buf = r.read(min(READ, span - got))
                    if not buf:
                        break
                    fh.write(buf)
                    got += len(buf)
                    prog.add(len(buf))
            if got == span:
                return True
        except Exception:
            pass
        if got:  # roll back the partial contribution before retrying
            prog.add(-got)
        time.sleep(1.5 * (attempt + 1))
    return False


def download(url: str, dest: str, conns: int, chunk: int, label: str) -> None:
    total = remote_size(url)
    if total <= 0:
        raise RuntimeError("cannot determine remote size")

    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print(f"  {label} already complete ({human(total)})")
        return

    state_path = dest + ".parts.json"
    done: set[int] = set()
    if os.path.exists(state_path):
        try:
            saved = json.load(open(state_path))
            if saved.get("total") == total:
                done = set(saved.get("done") or [])
        except Exception:
            done = set()

    parts = [
        (i, i * chunk, min((i + 1) * chunk, total))
        for i in range((total + chunk - 1) // chunk)
    ]

    # Salvage a partial file left by an earlier sequential download: its bytes
    # from 0 are valid, so any chunk it fully covers counts as done.
    if not done and os.path.exists(dest):
        have = os.path.getsize(dest)
        if 0 < have < total:
            salvaged = [i for i, s, e in parts if e <= have]
            if salvaged:
                done = set(salvaged)
                print(f"  reusing {human(have)} already downloaded ({len(salvaged)} chunk(s))")

    # Preallocate so chunks can be written at their final offsets.
    with open(dest, "r+b" if os.path.exists(dest) else "wb") as fh:
        fh.truncate(total)
    todo = [p for p in parts if p[0] not in done]

    already = sum(e - s for i, s, e in parts if i in done)
    prog = Progress(total, label)
    prog.done = already
    print(f"  {label} size {human(total)} | {len(todo)} chunk(s) left | {conns} connections")

    lock = threading.Lock()

    def worker(part: tuple[int, int, int]) -> bool:
        idx, s, e = part
        ok = fetch_chunk(url, dest, s, e, prog)
        if ok:
            with lock:
                done.add(idx)
                json.dump({"total": total, "done": sorted(done)}, open(state_path, "w"))
        return ok

    failed = []
    with ThreadPoolExecutor(max_workers=conns) as pool:
        for part, ok in zip(todo, pool.map(worker, todo)):
            if not ok:
                failed.append(part[0])

    prog.finish()
    if failed:
        raise RuntimeError(f"{len(failed)} chunk(s) failed: {failed[:8]}")
    if os.path.getsize(dest) != total:
        raise RuntimeError("size mismatch after download")
    os.remove(state_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report missing files")
    ap.add_argument("--comfy", default=COMFY_ROOT, help="ComfyUI root directory")
    ap.add_argument("--conns", type=int, default=DEFAULT_CONNS, help="parallel connections")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="chunk size in bytes")
    args = ap.parse_args()

    if not os.path.isdir(args.comfy):
        print(f"ComfyUI root not found: {args.comfy}", file=sys.stderr)
        return 2

    missing = 0
    for repo, path, folder, name in MODELS:
        target = os.path.join(args.comfy, "models", folder, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        present = os.path.exists(target) and os.path.getsize(target) > 0
        print(f"[{folder}] {name}")
        if args.check:
            print("  present" if present else "  MISSING")
            missing += 0 if present else 1
            continue
        try:
            download(url_for(repo, path), target, args.conns, args.chunk, name)
        except Exception as exc:  # keep going; the next run resumes
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            missing += 1
    if args.check and missing:
        print(f"\n{missing} file(s) missing")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
