"""Survey the saved ComfyUI workflows: format, models wanted, nodes available.

More accurate than the first pass:
  * node types are parsed from NODE_CLASS_MAPPINGS dict keys via AST, covering
    nodes.py + comfy_extras/ + custom_nodes/  (the first version missed comfy_extras and
    reported built-ins such as ImageUpscaleWithModel / Canny as missing)
  * model paths are collected from ANY widget value that ends in a model extension
    (the first version took widgets_values[0] and so mistook strength numbers like 0.85
    for file names)

    python comfyui/survey_workflows.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import paths

COMFY = paths.COMFY_ROOT
WF_DIR = paths.WORKFLOWS_DIR
ZIP_PATH = WF_DIR / "入门工作流" / "ComfyUI_Workflows.zip"
MODELS = paths.MODELS_DIR
MODEL_EXT = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".npz", ".pkl", ".onnx"}

# Lite-graph-only nodes: they draw nothing and compute nothing, safe to drop on conversion.
UI_ONLY = {"Note", "Reroute", "PrimitiveNode", "MarkdownNote", "Anything", "Constant"}


def local_models() -> set[str]:
    out = set()
    for p in MODELS.rglob("*"):
        if p.is_file() and p.suffix.lower() in MODEL_EXT:
            out.add(p.relative_to(MODELS).as_posix().lower())
            out.add(p.name.lower())
    return out


NODE_ID_RE = re.compile(r'node_id\s*=\s*"([A-Za-z_][\w\- ]*)"')


def node_class_keys(path: Path) -> set[str]:
    """Registered node names in one file, via EITHER declaration style.

    1. NODE_CLASS_MAPPINGS = {"Name": Cls, ...}   custom nodes + some extras
    2. node_id="Name"                             NEW V2 NodeRegistry style used by
       comfy_extras in current ComfyUI - the first version of this script only looked
       for (1) and therefore wrongly reported built-ins (Canny, ImageUpscaleWithModel,
       UpscaleModelLoader, CLIPTextEncodeSDXL) as missing. Always parse both.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return set()
    found = set(NODE_ID_RE.findall(text))
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any("CLASS_MAPPINGS" in t.upper() for t in targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Dict):
                for k in sub.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        found.add(k.value)
    return found


def installed_nodes_live() -> set[str] | None:
    """Authoritative node list from the running server, or None if it is not up.

    The AST scan below is fragile: comfyui_controlnet_aux registers its nodes from
    node_packages/**/subfolders and via dynamic maps, so a static grep of top-level *.py
    wrongly reported DWPreprocessor / OpenposePreprocessor / ShufflePreprocessor as missing
    even though they are installed and working (pose.json ran end to end). When the server is
    up, /object_info is ground truth - always prefer it and keep the scan only as the offline
    fallback.
    """
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=180) as r:
            return set(json.load(r))
    except Exception:
        return None


def installed_nodes() -> tuple[set[str], dict[str, str]]:
    types: set[str] = set()
    origin: dict[str, str] = {}
    files = [COMFY / "nodes.py"] + sorted((COMFY / "comfy_extras").glob("*.py"))
    for d in sorted((COMFY / "custom_nodes").glob("*")):
        if d.is_dir():
            files += [d / "__init__.py"] + sorted(d.rglob("*.py"))
    for f in files:
        if f.is_file():
            got = node_class_keys(f)
            for t in got:
                origin.setdefault(t, str(f.relative_to(COMFY)).replace("\\", "/"))
            types |= got
    return types, origin


def load_ui(raw: str):
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("nodes"), list) and "links" in d else None


def harvest():
    found = []
    for p in sorted(WF_DIR.rglob("*.json")):
        ui = load_ui(p.read_text(encoding="utf-8", errors="ignore"))
        if ui:
            found.append((p.stem, str(p.relative_to(WF_DIR)), ui, p))
    if ZIP_PATH.is_file():
        with zipfile.ZipFile(ZIP_PATH) as z:
            for n in z.namelist():
                if not n.lower().endswith(".json") or "__MACOSX" in n:
                    continue
                try:
                    ui = load_ui(z.read(n).decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                if ui and not any(f[0] == Path(n).stem for f in found):
                    found.append((Path(n).stem, f"ZIP:{n}", ui, None))
    return found


def looks_like_model(v) -> bool:
    return isinstance(v, str) and Path(v).suffix.lower() in MODEL_EXT


def main() -> None:
    models = local_models()
    have, origin = installed_nodes()
    live = installed_nodes_live()
    if live is not None:
        have |= live  # union: server is authoritative, scan adds nothing it lacks
        src_note = f"实时 /object_info {len(live)} 个 + 静态扫描，合计 {len(have)}"
    else:
        src_note = f"仅静态扫描 {len(have)}（服务未运行，Preprocessor 类可能误报缺失）"
    rows = []
    for name, src, ui, _ in harvest():
        wanted, node_types, ui_only = set(), [], set()
        for n in ui["nodes"]:
            t = str(n.get("type", "?"))
            node_types.append(t)
            if t in UI_ONLY:
                ui_only.add(t)
            for wv in n.get("widgets_values") or []:
                if looks_like_model(wv):
                    wanted.add(Path(str(wv)).as_posix())
        missing_models = sorted(m for m in wanted if m.lower() not in models and Path(m).name.lower() not in models)
        # resolve deprecated/renamed ids too, so "ConditioningAverage " is not a false missing
        missing_nodes = sorted({t for t in node_types
                                if t not in have and t.strip() not in {h.strip() for h in have} and t not in UI_ONLY})
        rows.append((name, src, len(node_types), sorted(wanted), missing_models, missing_nodes, ui_only))

    print(f"UI 格式工作流: {len(rows)} 个 | 节点判定依据: {src_note} | 本地模型文件: {len(models)//2}\n")
    runnable, near = [], []
    for name, src, n, wanted, mm, mn, uio in sorted(rows, key=lambda r: (len(r[4]) + len(r[5]), r[0])):
        if not mm and not mn:
            runnable.append(name)
            tag = "OK"
        elif not mn and len(mm) == 1 and mm[0].startswith("vae-ft-mse"):
            near.append(name)
            tag = "NEAR"
        else:
            tag = "NO"
        print(f"[{tag:4}] {name}  ({n} nodes)")
        print(f"       需要模型: {', '.join(wanted) or '-'}")
        if mm:
            print(f"       缺模型  : {', '.join(mm)}")
        if mn:
            print(f"       缺节点  : {', '.join(mn)}")
        if uio:
            print(f"       纯UI节点(转换时丢弃): {', '.join(sorted(uio))}")

    print(f"\n>>> 完全可直接跑 ({len(runnable)}): {', '.join(runnable) or '无'}")
    print(f">>> 只差独立 VAE 文件 ({len(near)}): {', '.join(near)}")
    print("    vae-ft-mse-840000 只有 ~340MB，hf-mirror 可下；或改成不接 vaename 用 ckpt 自带 VAE")


if __name__ == "__main__":
    main()
