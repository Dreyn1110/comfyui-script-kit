# -*- coding: utf-8 -*-
r"""本机 ComfyUI 统一入口（一条命令出图 / 超分 / 改图）。

取代以前的 6 个脚本：run_t2i.py, run_hires.py, run_qwen_edit.py,
run_anime_t2i.py, run_anime_hiresfix.py, upscale_demo.py。

**永远用 ComfyUI 自带的解释器跑**（系统的 python 没装 pyyaml 等包）：
    D:\Al\ComfyUI-aki-v3.2\python\python.exe comfyui\dsh.py <命令> [参数]

命令：
    t2i       动漫文生图（Counterfeit 底模 + 2 个 LoRA），默认 512x768
    hires     同上 + latent 2x 重采样补细节（约 1024x1536，实测 6GB 可跑，~27s）
    upscale   把一张已有图用 4x 放大模型放大（--src 必传）
    edit      Qwen-Image-Edit 改一张图（--src + --prompt 必传，~105-145s）
    sd        直接跑一个 SD1.5 工作流 json（t2i_workflow.json / t2i_anime_hires.json）
    wf        跑**网页里保存的** UI 格式工作流（自动转 API 格式，见 ui2api.py）

例子：
    dsh.py t2i                          # 用内置默认提示词出一张动漫图
    dsh.py t2i --prompt "1girl, ..." --seed 123
    dsh.py t2i --upscale                # 出图后自动 4x 放大（串联，中间会释放显存）
    dsh.py hires --upscale -o 最终图.png
    dsh.py upscale --src anime_高级图.png
    dsh.py edit  --src 笔记.png --prompt "把纸张换成淡黄色，字迹不变"
    dsh.py wf --json clip_skip          # 名字会在 comfyui/、工作区根、ComfyUI workflows 目录里找
    dsh.py wf --json upscale_by_model --dry                # 只看转换结果，不出图
    dsh.py wf --json word_weighting --set text="1girl, ..." --seed 7

设计约定（都是这台机器踩过的坑，别再改回去）：
  - 输出目录由 __file__ 推导 = 真实工作区，绝不写死绝对路径。
  - 轮询一定先判 status_str=="error"，否则节点报错会被伪装成"超时"。
  - 取放图一律走 /upload/image + /view，不往 ComfyUI 安装目录 copy 文件。
  - 服务没起时默认自动拉起 start_comfyui.ps1；用 --no-autostart 关掉。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import paths  # noqa: E402  统一解析安装目录 / API / 输出目录（见 paths.py）

API = paths.API
HERE = Path(__file__).resolve().parent            # .../comfyui
WS = paths.OUT_DIR                                # ← 输出默认落这里
START_PS = HERE / "start_comfyui.ps1"
EDIT_JSON = HERE / "qwen-edit-2511-quality.json"
# 用户在 ComfyUI 网页里保存的工作流默认落这里（UI 格式，要经 ui2api 转换）
WF_DIR = paths.WORKFLOWS_DIR

# ---- 动漫工作流默认值（集中一处，改一次即可）----
ANIME_CKPT = "Counterfeit-V3.0_fix_fp16.safetensors"
LORA_DETAIL = ("add_detail.safetensors", 0.6)
LORA_NOISE = ("epi_noiseoffset.safetensors", 0.4)
ANIME_POS = (
    "masterpiece, best quality, ultra-detailed, 8k, 1girl, "
    "beautiful detailed eyes, long flowing silver hair, elegant black gothic dress, "
    "standing in a moonlit japanese garden, cherry blossom petals floating in the air, "
    "soft rim lighting, cinematic lighting, glowing fireflies, dramatic shadows, "
    "depth of field, detailed background, intricate details"
)
ANIME_NEG = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)
UPSCALE_MODEL = "4x-AnimeSharp.pth"


def utf8_console():
    """Force UTF-8 on BOTH stdout and stderr.

    Windows PowerShell 5.1 hands python a GBK console stream, so every Chinese message on
    stderr (all of sys.exit's output, argparse errors) comes out as mojibake. Configuring
    only stdout - as this file used to - fixes the progress lines and leaves the one message
    the user most needs readable in the worst encoding possible.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.encoding and stream.encoding.lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def find_workflow(spec):
    """Resolve a workflow given as a path, a bare name, or a name under ComfyUI's folders.

    Shared by dsh.py wf and ui2api.py on purpose: two entries each with their own lookup
    drift apart the first time one is fixed. ComfyUI's saved workflows sit in nested folders
    (入门工作流\\basic\\...), so a bare name is searched recursively; several matches are
    reported rather than guessed at. Returns a Path, or None when nothing matches.
    """
    p = Path(spec)
    for cand in (p, HERE / spec, WS / spec):
        if cand.is_file():
            return cand
    hits = sorted({f for f in WF_DIR.rglob("*.json")
                   if f.stem == p.stem and "__pycache__" not in f.parts}) if WF_DIR.is_dir() else []
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise AmbiguousWorkflow(p.stem, hits)
    return None


class AmbiguousWorkflow(RuntimeError):
    def __init__(self, stem, hits):
        super().__init__(f"'{stem}' 有多个同名工作流，请给完整路径：\n  " + "\n  ".join(str(h) for h in hits))


# ---------------------------------------------------------------- HTTP 层
def _json_request(url, payload=None, timeout=120):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(
        url, method="POST" if payload is not None else "GET", data=data, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        # /free 等端点会返回 200 + 空 body，直接 json.load 会抛
        # "Expecting value: line 1 column 1"。空响应视为成功、返回 None。
        return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}\n{body[:2000]}") from None


def ping(timeout=4):
    try:
        _json_request(API + "/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


def ensure_server(autostart=True, wait=200):
    if ping():
        return
    if not autostart:
        sys.exit("ComfyUI 没在跑。手动启动：\n  powershell -ExecutionPolicy Bypass -File comfyui\\start_comfyui.ps1")
    if not START_PS.exists():
        sys.exit(f"找不到启动脚本: {START_PS}")
    print("ComfyUI 未运行，正在自动启动（首次加载可能要 1-2 分钟）…")
    # CREATE_NO_WINDOW(0x08000000)：后台跑、不闪控制台窗。
    # 绝不能用 DETACHED_PROCESS(0x8)：受限沙箱下子进程会被静默丢弃 ——
    # Popen 正常返回、wait() 拿到 rc=0，但命令根本没执行过（已实测复现）。
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000
    subprocess.Popen(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(START_PS)], **kwargs
    )
    deadline = time.time() + wait
    while time.time() < deadline:
        time.sleep(2)
        if ping():
            print("ComfyUI 已就绪。")
            return
    sys.exit("ComfyUI 启动超时，请手动跑 start_comfyui.ps1 看日志。")


def upload_image(path):
    """POST /upload/image，返回 ComfyUI 侧存储名。"""
    boundary = "----dsh" + uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    blob = Path(path).read_bytes()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        blob,
        f"\r\n--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        API + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        info = json.load(r)
    stored = info.get("name") or name
    sub = info.get("subfolder") or ""
    return f"{sub}/{stored}" if sub else stored


def fetch_image(ref, dest):
    """GET /view 取回一张输出图，写到 dest。"""
    q = urllib.parse.urlencode({
        "filename": ref["filename"],
        "subfolder": ref.get("subfolder", ""),
        "type": ref.get("type", "output"),
    })
    with urllib.request.urlopen(f"{API}/view?{q}", timeout=300) as r:
        data = r.read()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(data)
    return dest, len(data)


def submit_and_wait(workflow, timeout=400, label="生成"):
    client_id = uuid.uuid4().hex
    try:
        res = _json_request(API + "/prompt", {"prompt": workflow, "client_id": client_id})
    except RuntimeError as e:
        sys.exit(f"提交失败：\n{e}")
    pid = res["prompt_id"]
    print(f"[{label}] prompt_id={pid}，等待出图…")
    t0 = time.time()
    while True:
        time.sleep(2.5)
        hist = _json_request(API + f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            # 先判错：失败时可能永远没有 outputs，也会没有 completed。
            if status.get("status_str") == "error":
                sys.exit(f"[{label}] 出图失败：{json.dumps(status, ensure_ascii=False)[:2000]}")
            imgs = [i for o in (entry.get("outputs") or {}).values() for i in o.get("images", [])]
            if imgs:
                el = time.time() - t0
                print(f"[{label}] 完成，用时 {el:.0f}s")
                return imgs
            # 已完成却没有图片（比如工作流没有 SaveImage）：直接说清楚，
            # 别让它空转到超时、又被当成"超时"这个假原因。
            if status.get("completed"):
                sys.exit(f"[{label}] 任务完成但没有输出图片（工作流缺 SaveImage？）。outputs={json.dumps(entry.get('outputs'), ensure_ascii=False)[:500]}")
        if time.time() - t0 > timeout:
            sys.exit(f"[{label}] 超时（{timeout}s）。服务还在吗？节点是否报错？")


def free_vram():
    """让服务端在处理下一个 prompt 前卸载模型/释放内存。

    注意 /free 的实际契约（server.py:1193-1201）：只认
    {"unload_models": true, "free_memory": true}，是给队列打 flag，
    作用于**下一个**排队的 prompt，不是立即清空。
    它对任何 body 都返回 200（未知键被 .get 默认 False 静默吞掉），
    所以不能用 200 判断"生效了" —— 曾经这里发的是 cache_lifecycle，
    返回 200 但等于没调用。这里失败也只打一行提示，不打断出图。
    """
    try:
        _json_request(API + "/free",
                      {"unload_models": True, "free_memory": True}, timeout=30)
        print("  (已请求卸载模型，将在下一步生效)")
    except Exception as e:
        print(f"  (跳过显存释放: {e})")


# ---------------------------------------------------------------- 工作流构造
def _anime_prelude(seed, pos, neg):
    """底模 + 两个 LoRA + 正负提示词（t2i 和 hires 共用，曾经两处逐字复制）。"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ANIME_CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "lora_name": LORA_DETAIL[0], "strength_model": LORA_DETAIL[1], "strength_clip": LORA_DETAIL[1],
            "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "LoraLoader", "inputs": {
            "lora_name": LORA_NOISE[0], "strength_model": LORA_NOISE[1], "strength_clip": LORA_NOISE[1],
            "model": ["2", 0], "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["3", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["3", 1]}},
    }


def build_anime_t2i(seed, w, h, pos, neg):
    wf = _anime_prelude(seed, pos, neg)
    wf["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    wf["7"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": 28, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 1.0, "model": ["3", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "anime", "images": ["8", 0]}}
    return wf


def build_anime_hires(seed, w, h, pos, neg, denoise):
    wf = _anime_prelude(seed, pos, neg)
    wf["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    # 第一阶段：低分辨率构图
    wf["7"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": 28, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 1.0, "model": ["3", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}}
    # latent 2x
    wf["8"] = {"class_type": "LatentUpscale", "inputs": {
        "samples": ["7", 0], "upscale_method": "nearest-exact", "width": w * 2, "height": h * 2, "crop": "disabled"}}
    # 第二阶段：denoise 重采样补细节
    wf["9"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": 20, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": denoise, "model": ["3", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["8", 0]}}
    wf["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}}
    wf["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "anime_hires", "images": ["10", 0]}}
    return wf


def build_upscale(image_name, model):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "anime_x4", "images": ["3", 0]}},
    }


def fill_edit_template(template, prompt, image_name, seed):
    """把 qwen-edit json 里的 {{image}}/{{prompt}}/{{seed}} 占位符替换掉。"""
    def walk(v):
        if isinstance(v, str):
            if v == "{{image}}":
                return image_name
            if v == "{{seed}}":
                return seed
            return v.replace("{{prompt}}", prompt)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v
    return {nid: walk(node) for nid, node in template.items()}


# ---------------------------------------------------------------- 命令
def _resolve_out(out_arg, default_name):
    if out_arg:
        p = Path(out_arg)
        return p if p.is_absolute() else WS / out_arg
    return WS / default_name


def cmd_t2i(args):
    ensure_server(args.autostart)
    seed = args.seed or random.randint(1, 2**31)
    wf = build_anime_t2i(seed, args.w, args.h, args.prompt or ANIME_POS, args.neg or ANIME_NEG)
    print(f"[t2i] seed={seed} {args.w}x{args.h}")
    imgs = submit_and_wait(wf, args.timeout, "t2i")
    dest, n = fetch_image(imgs[0], _resolve_out(args.out, "anime_高级图.png"))
    print(f"→ {dest}  ({n} 字节)")
    if args.upscale:
        free_vram()
        _do_upscale(str(dest), str(dest).rsplit(".", 1)[0] + "_4x.png", args.upscale_model, args.timeout)


def cmd_hires(args):
    ensure_server(args.autostart)
    seed = args.seed or random.randint(1, 2**31)
    wf = build_anime_hires(seed, args.w, args.h, args.prompt or ANIME_POS, args.neg or ANIME_NEG, args.denoise)
    print(f"[hires] seed={seed} 底图 {args.w}x{args.h} → {args.w*2}x{args.h*2}  denoise={args.denoise}")
    imgs = submit_and_wait(wf, args.timeout, "hires")
    dest, n = fetch_image(imgs[0], _resolve_out(args.out, "anime_高清.png"))
    print(f"→ {dest}  ({n} 字节)")
    if args.upscale:
        free_vram()
        _do_upscale(str(dest), str(dest).rsplit(".", 1)[0] + "_4x.png", args.upscale_model, args.timeout)


def _do_upscale(src, out, model, timeout):
    if not Path(src).is_file():
        sys.exit(f"输入图不存在: {src}")
    name = upload_image(src)
    print(f"[upscale] 已上传 {src} → {name}")
    imgs = submit_and_wait(build_upscale(name, model), timeout, "upscale")
    dest, n = fetch_image(imgs[0], Path(out) if Path(out).is_absolute() else WS / out)
    print(f"→ {dest}  ({n} 字节)")


def cmd_upscale(args):
    ensure_server(args.autostart)
    out = str(_resolve_out(args.out, "anime_高级图_4x.png"))
    _do_upscale(args.src, out, args.upscale_model, args.timeout)


def cmd_edit(args):
    ensure_server(args.autostart)
    if not EDIT_JSON.exists():
        sys.exit(f"缺编辑工作流: {EDIT_JSON}")
    if not Path(args.src).is_file():
        sys.exit(f"输入图不存在: {args.src}")
    template = json.loads(EDIT_JSON.read_text(encoding="utf-8"))
    name = upload_image(args.src)
    print(f"[edit] 已上传 {args.src} → {name}")
    seed = args.seed or random.randint(1, 2**31)
    wf = fill_edit_template(template, args.prompt, name, seed)
    print(f"[edit] seed={seed}  (Qwen-Image-Edit-2511，约 105-145s)")
    imgs = submit_and_wait(wf, args.timeout, "edit")
    dest, n = fetch_image(imgs[0], _resolve_out(args.out, f"QwenEdit_{seed}.png"))
    print(f"→ {dest}  ({n} 字节)")


def cmd_sd(args):
    ensure_server(args.autostart)
    jf = Path(args.json)
    if not jf.is_absolute():
        # 先按 comfyui/ 找，再按工作区根找（两个 json 原本就在工作区根）
        for cand in (HERE / args.json, WS / args.json):
            if cand.exists():
                jf = cand
                break
    if not jf.exists():
        sys.exit(f"找不到工作流 json: {args.json}")
    wf = json.loads(jf.read_text(encoding="utf-8"))
    if args.seed:
        for node in wf.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = args.seed
    imgs = submit_and_wait(wf, args.timeout, "sd")
    base = _resolve_out(args.out, f"{args.prefix}.png")
    stem, ext = str(base).rsplit(".", 1)
    for k, ref in enumerate(imgs):
        # 单图用干净文件名；多批才加 _1/_2 后缀，避免双后缀。
        dest = base if len(imgs) == 1 else f"{stem}_{k}.{ext}"
        _, n = fetch_image(ref, dest)
        print(f"→ {dest}  ({n} 字节)")


def cmd_wf(args):
    """跑一个**网页里保存的** UI 格式工作流（含 nodes/links 那种）。

    /prompt 只吃 API 格式，所以先交给 ui2api 转换；转换需要 /object_info，
    因此这里必须先确保服务在跑。
    """
    ensure_server(args.autostart)
    import ui2api
    try:
        jf = find_workflow(args.json)
    except AmbiguousWorkflow as e:
        sys.exit(str(e))
    if jf is None:
        sys.exit(f"找不到工作流 json: {args.json}\n"
                 f"（找过 comfyui/、工作区根、{WF_DIR}；也可以直接给绝对路径）")
    ui = json.loads(jf.read_text(encoding="utf-8"))
    if not isinstance(ui, dict) or "nodes" not in ui:
        sys.exit(f"{jf.name} 顶层没有 nodes —— 已经是 API 格式，用： dsh.py sd --json {args.json}")
    notes: list[str] = []
    try:
        api = ui2api.convert(ui, notes)
    except ui2api.ServerGone as e:
        sys.exit(f"转换中断：{e}")
    for kv in args.sets:
        ui2api.apply_sets(api, [kv], notes)
    if args.seed:
        for node in api.values():
            if "seed" in node["inputs"]:
                node["inputs"]["seed"] = args.seed
    print(f"[wf] {jf.name}: {len(ui.get('nodes') or [])} UI 节点 -> {len(api)} API 节点")
    for line in notes:
        print(line)
    if args.dry:
        print(json.dumps(api, ensure_ascii=False, indent=2))
        return
    imgs = submit_and_wait(api, args.timeout, "wf")
    ui2api.save_images(imgs, args.out)


# ---------------------------------------------------------------- 解析
def add_shared(p):
    p.add_argument("--seed", type=int, default=0, help="随机种子（0=随机）")
    p.add_argument("-o", "--out", default=None, help="输出路径，默认写进工作区根目录")
    p.add_argument("--timeout", type=int, default=400, help="等待出图上限秒数")
    p.add_argument("--upscale", action="store_true", help="出图后自动再 4x 放大（串联）")
    p.add_argument("--upscale-model", default=UPSCALE_MODEL, help=f"放大模型，默认 {UPSCALE_MODEL}")
    p.add_argument("--no-autostart", dest="autostart", action="store_false", help="服务没起时不自动拉起，只报错")
    p.set_defaults(autostart=True)


def build_parser():
    ap = argparse.ArgumentParser(prog="dsh.py", description="本机 ComfyUI 统一入口",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("t2i", help="动漫文生图")
    a.add_argument("--prompt", default=None, help="正向提示词（不给用内置默认）")
    a.add_argument("--neg", default=None, help="负向提示词")
    a.add_argument("--w", type=int, default=512)
    a.add_argument("--h", type=int, default=768)
    add_shared(a)
    a.set_defaults(func=cmd_t2i)

    b = sub.add_parser("hires", help="动漫文生图 + 高清修复")
    b.add_argument("--prompt", default=None)
    b.add_argument("--neg", default=None)
    b.add_argument("--w", type=int, default=512)
    b.add_argument("--h", type=int, default=768)
    b.add_argument("--denoise", type=float, default=0.5, help="第二阶段重绘强度，默认 0.5")
    add_shared(b)
    b.set_defaults(func=cmd_hires)

    c = sub.add_parser("upscale", help="把一张已有图 4x 放大")
    c.add_argument("--src", required=True, help="要放大的图片路径（必传）")
    c.add_argument("-o", "--out", default=None)
    c.add_argument("--upscale-model", default=UPSCALE_MODEL)
    c.add_argument("--timeout", type=int, default=180)
    c.add_argument("--no-autostart", dest="autostart", action="store_false")
    c.set_defaults(autostart=True, func=cmd_upscale)

    d = sub.add_parser("edit", help="Qwen-Image-Edit 改一张图")
    d.add_argument("--src", required=True, help="要编辑的图片路径（必传）")
    d.add_argument("--prompt", required=True, help="编辑指令（必传）")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("-o", "--out", default=None)
    d.add_argument("--timeout", type=int, default=400)
    d.add_argument("--no-autostart", dest="autostart", action="store_false")
    d.set_defaults(autostart=True, func=cmd_edit)

    e = sub.add_parser("sd", help="直接跑一个 SD1.5 工作流 json")
    e.add_argument("--json", default="t2i_workflow.json", help="工作流 json 文件名")
    e.add_argument("--prefix", default="dsh_sd", help="输出文件名前缀（多批图会加 _0/_1 后缀）")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("-o", "--out", default=None)
    e.add_argument("--timeout", type=int, default=600)
    e.add_argument("--no-autostart", dest="autostart", action="store_false")
    e.set_defaults(autostart=True, func=cmd_sd)

    f = sub.add_parser("wf", help="跑网页保存的 UI 格式工作流（自动转 API 格式）")
    f.add_argument("--json", required=True, help="工作流文件名或路径（会在 comfyui/、工作区、ComfyUI workflows 里找）")
    f.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE",
                   help="覆盖输入，可重复。text/prompt 类有别名，见 ui2api")
    f.add_argument("--seed", type=int, default=0, help="覆盖所有带 seed 的节点（0=用文件里的值）")
    f.add_argument("--dry", action="store_true", help="只打印转换结果，不出图")
    f.add_argument("-o", "--out", default=None, help="输出图片路径（多输出时自动加 _1/_2）")
    f.add_argument("--timeout", type=int, default=600)
    f.add_argument("--no-autostart", dest="autostart", action="store_false")
    f.set_defaults(autostart=True, func=cmd_wf)

    return ap


if __name__ == "__main__":
    # cmd_wf imports ui2api, which imports dsh. Run as a script, this module is __main__ and
    # has no "dsh" entry in sys.modules, so that import would load a SECOND copy with its own
    # globals. Registering the running module first makes ui2api reuse it.
    sys.modules.setdefault("dsh", sys.modules[__name__])
    utf8_console()
    args = build_parser().parse_args()
    args.func(args)
