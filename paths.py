"""Central path + endpoint resolution.

Every script used to hard-code D:\\Al\\ComfyUI-aki-v3.2, which made the toolkit
unusable on any other machine. Paths now resolve from environment variables,
falling back to that same default so existing local usage is unchanged.

    COMFYUI_HOME   ComfyUI install root (the folder containing ComfyUI/ and python/)
    COMFYUI_API    API base URL              (default http://127.0.0.1:8188)
    COMFYUI_OUT    where generated images go (default: this folder's parent)

Set them once in your shell profile, or pass --comfy-home to a script that supports it.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_HOME = Path(r"D:\Al\ComfyUI-aki-v3.2")


def _home() -> Path:
    """Install root: $COMFYUI_HOME, else the historical default, else a clear error."""
    raw = os.environ.get("COMFYUI_HOME")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_HOME


COMFYUI_HOME: Path = _home()
COMFY_ROOT: Path = COMFYUI_HOME / "ComfyUI"
MODELS_DIR: Path = COMFY_ROOT / "models"
WORKFLOWS_DIR: Path = COMFY_ROOT / "user" / "default" / "workflows"
PYTHON_EXE: Path = COMFYUI_HOME / "python" / "python.exe"

API: str = os.environ.get("COMFYUI_API", "http://127.0.0.1:8188").rstrip("/")

# Generated images land here by default (the folder that owns this package).
OUT_DIR: Path = Path(os.environ.get("COMFYUI_OUT") or HERE.parent)


def require_home(*, hint: str = "") -> None:
    """Fail loudly with an actionable message instead of a bare FileNotFoundError."""
    if not COMFY_ROOT.is_dir():
        raise SystemExit(
            f"找不到 ComfyUI 安装目录：{COMFY_ROOT}\n"
            "请设置环境变量 COMFYUI_HOME 指向安装根目录（其下应有 ComfyUI\\ 和 python\\）。\n"
            "Windows PowerShell 示例：\n"
            '  $env:COMFYUI_HOME = "C:\\ComfyUI"\n'
            + (f"\n{hint}" if hint else "")
        )
