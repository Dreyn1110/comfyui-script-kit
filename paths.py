"""Central path + endpoint resolution.

Paths resolve from environment variables first, then from a short list of common
install locations. Nothing is hard-coded to one machine, and a missing install
produces an actionable message instead of an opaque FileNotFoundError.

    COMFYUI_HOME   ComfyUI install root (the folder containing ComfyUI/ and python/)
    COMFYUI_API    API base URL              (default http://127.0.0.1:8188)
    COMFYUI_OUT    where generated images go (default: this folder's parent)

Windows PowerShell:
    $env:COMFYUI_HOME = "C:\\ComfyUI"

Linux / macOS:
    export COMFYUI_HOME="$HOME/ComfyUI"
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Tried in order when COMFYUI_HOME is unset. Set the env var to skip this list.
CANDIDATES = (
    Path.home() / "ComfyUI",
    Path("C:/ComfyUI"),
    Path("/opt/ComfyUI"),
    Path("/usr/local/ComfyUI"),
)


def candidate_homes() -> list[Path]:
    """Install roots to try, in order. The env var wins outright when set.

    A gitignored ``comfyui.local.json`` ({"home": "..."}) is read next, so a local
    checkout can remember its install path without an environment variable and
    without that path ever reaching git.
    """
    raw = os.environ.get("COMFYUI_HOME")
    if raw:
        return [Path(raw).expanduser()]
    local = HERE / "comfyui.local.json"
    if local.is_file():
        try:
            # utf-8-sig, not utf-8: Windows PowerShell 5.1's `Out-File -Encoding utf8`
            # writes a BOM, and json.loads() rejects a leading BOM outright.
            data = json.loads(local.read_text(encoding="utf-8-sig"))
            home = data.get("home")
            if home:
                return [Path(home).expanduser()]
        except (OSError, ValueError):
            pass
    return list(CANDIDATES)


def _first_existing_home() -> Path:
    """Pick the first candidate that looks like a real install.

    A directory counts only when it contains ComfyUI/, so a stray empty C:/ComfyUI
    cannot win over a better candidate. When nothing matches, return the first
    candidate so the error message names a sensible path.
    """
    cands = candidate_homes()
    for c in cands:
        if (c / "ComfyUI").is_dir():
            return c
    return cands[0]


COMFYUI_HOME: Path = _first_existing_home()
COMFY_ROOT: Path = COMFYUI_HOME / "ComfyUI"
MODELS_DIR: Path = COMFY_ROOT / "models"
WORKFLOWS_DIR: Path = COMFY_ROOT / "user" / "default" / "workflows"
PYTHON_EXE: Path = COMFYUI_HOME / "python" / "python.exe"

API: str = os.environ.get("COMFYUI_API", "http://127.0.0.1:8188").rstrip("/")

# Generated images land here by default (the folder that owns this package).
OUT_DIR: Path = Path(os.environ.get("COMFYUI_OUT") or HERE.parent)


def require_home(*, hint: str = "") -> None:
    """Fail loudly instead of raising a bare FileNotFoundError deep in an HTTP call."""
    if not COMFY_ROOT.is_dir():
        searchable = "\n".join(f"  {p}" for p in candidate_homes())
        raise SystemExit(
            f"ComfyUI installation not found (looked for {COMFY_ROOT}).\n"
            "Set COMFYUI_HOME to the folder that contains ComfyUI/ and python/.\n"
            f"Currently trying:\n{searchable}\n\n"
            'Windows PowerShell:  $env:COMFYUI_HOME = "C:\\ComfyUI"\n'
            'Linux / macOS:       export COMFYUI_HOME="$HOME/ComfyUI"'
            + (f"\n\n{hint}" if hint else "")
        )
