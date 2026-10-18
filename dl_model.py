"""Generic hf-mirror model downloader with resume + verification.

The bundled PowerShell TLS stack is broken on this machine (SEC_E_NO_CREDENTIALS), so every
model fetch goes through Python urllib. Existing download_anime_models.py hard-codes its
own list; this one takes arguments so any model can be pulled without editing code.

    python comfyui/dl_model.py stabilityai/sd-vae-ft-mse-original \
        vae-ft-mse-840000-ema-pruned.safetensors --subdir vae
    python comfyui/dl_model.py <repo> <remote_file> --subdir checkpoints --as 本地文件名
    python comfyui/dl_model.py --list            # what the local install already has
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import paths

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
MODELS_DIR = paths.MODELS_DIR
CTX = ssl.create_default_context()


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def head(url: str):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
        return r.status, int(r.headers.get("Content-Length") or 0), r.headers.get("ETag", "")


def fetch(url: str, dest: Path, expect: int, tries: int = 6) -> bool:
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90, context=CTX) as r:
                code = r.getcode()
                if have and code != 206:  # server ignored Range: restart clean
                    print(f"  服务器不支持续传(code {code})，从头重下")
                    have = 0
                mode = "ab" if have and code == 206 else "wb"
                total = have + int(r.headers.get("Content-Length") or 0) if mode == "ab" else expect
                start, last, last_have = time.time(), time.time(), have
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1024 * 512)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 2:
                            spd = (have - last_have) / (time.time() - last) / 1048576
                            pct = 100 * have / max(total, 1)
                            print(f"  {have/1048576:.0f}/{total/1048576:.0f}MB ({pct:.0f}%) {spd:.1f}MB/s", flush=True)
                            last, last_have = time.time(), have
            if expect and have < expect:
                raise OSError(f"连接中断：{have}/{expect} 字节")
            tmp.replace(dest)
            print(f"  [OK] {dest.name}  {human(dest.stat().st_size)}")
            return True
        except Exception as e:
            print(f"  [重试 {attempt}/{tries}] {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(min(2 * attempt, 10))
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="从 hf-mirror 下载模型到 ComfyUI models 目录")
    ap.add_argument("repo", nargs="?", help="仓库，如 stabilityai/sd-vae-ft-mse-original")
    ap.add_argument("remote", nargs="?", help="仓库内文件名")
    ap.add_argument("--subdir", default="checkpoints", help="models/ 下的子目录，默认 checkpoints")
    ap.add_argument("--dest-root", default=None,
                    help="绝对根目录，覆盖 --subdir 的 models/ 前缀。"
                         "用于把权重放到 models/ 之外（如 custom_nodes/comfyui_controlnet_aux/ckpts）")
    ap.add_argument("--as", dest="asname", default=None, help="保存到本地的文件名（默认同远端）")
    ap.add_argument("--url", default=None, help="直接给完整 URL（跳过镜像拼接）")
    ap.add_argument("--list", action="store_true", help="列出本机已有模型")
    args = ap.parse_args()

    if args.list:
        for p in sorted(MODELS_DIR.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}:
                print(f"{p.stat().st_size/1048576:>8.0f}MB  {p.relative_to(MODELS_DIR)}")
        return 0
    if not args.repo or not args.remote:
        ap.error("需要 repo 和 remote 两个参数（或用 --list）")

    fname = args.asname or args.remote
    dest_dir = Path(args.dest_root) if args.dest_root else MODELS_DIR / args.subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[跳过] {args.subdir}/{fname} 已存在 ({human(dest.stat().st_size)})")
        return 0

    url = args.url or f"{MIRROR}/{args.repo}/resolve/main/{args.remote}"
    print(f">> {url}\n   -> {dest}")
    try:
        status, size, _ = head(url)
        print(f"   HTTP {status}, {human(size) if size else '大小未知'}")
    except Exception as e:
        print(f"   HEAD 失败（继续试 GET）: {e}")
        size = 0
    ok = fetch(url, dest, size)
    if ok:
        # safetensors/pt 至少能读出 header，作为"文件没截断"的廉价校验
        try:
            with open(dest, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if 0 < n < 10_000_000:
                    meta = json.loads(f.read(n))
                    print(f"   张量数 {len(meta) - ('__metadata__' in meta)}")
        except Exception:
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
