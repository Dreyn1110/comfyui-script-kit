"""Insert the Qwen-Image-Edit workflow into the DSH image-gen plugin config.

Edits the user patch layer (~/.dsh/profiles/desktop/cordis.patch.yml) in place,
preserving all existing content and comments: the new keys are inserted into the
existing `- id: image-gen` entry's `config:` block.

A backup is written next to the file before any change. Re-running is safe: an
existing comfyuiWorkflows block for the same name is replaced, not duplicated.

    python install_workflow.py            # apply
    python install_workflow.py --check    # report only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

# yaml 只用于写后校验，不是必需依赖。系统的 python 3.13 没装 pyyaml，
# 以前顶层 import 会让脚本在 --check 阶段就 ModuleNotFoundError 崩掉。
try:
    import yaml
except ImportError:
    yaml = None

PATCH = Path.home() / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"
WORKFLOW = Path(__file__).with_name("qwen-edit-2509-handwriting.json")
ENTRY_ID = "image-gen"
WORKFLOW_NAME = "qwen-edit-2509"
BASE_URL = "http://127.0.0.1:8188"
TIMEOUT_MS = 1800000  # 30 min; plugin default is only 5 min


def indent_block(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else "" for line in text.splitlines())


def build_block(workflow_json: str) -> str:
    """The YAML lines to insert under the image-gen entry's config: block."""
    # Compact but still readable: re-dump with 2-space indent, keep key order.
    compact = json.dumps(json.loads(workflow_json), ensure_ascii=False, indent=2)
    return "\n".join(
        [
            f"    comfyuiBaseURL: {BASE_URL}",
            f"    comfyuiActiveWorkflow: {WORKFLOW_NAME}",
            f"    comfyuiTimeoutMs: {TIMEOUT_MS}",
            "    comfyuiWorkflows:",
            f"      - name: {WORKFLOW_NAME}",
            '        presetPrompt: ""',
            "        json: |",
            indent_block(compact, 10),
        ]
    )


def find_entry(lines: list[str]) -> tuple[int, int]:
    """Return (entry_start, config_line_index) for the image-gen entry."""
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^-\s+id:\s*{re.escape(ENTRY_ID)}\s*$", line):
            start = i
            break
    if start is None:
        raise SystemExit(f"could not find an entry with id: {ENTRY_ID} in {PATCH}")
    for j in range(start, len(lines)):
        # Stop if we run into the next top-level list entry.
        if j > start and re.match(r"^-\s+", lines[j]):
            raise SystemExit(f"entry {ENTRY_ID} has no config: block")
        if re.match(r"^\s+config:\s*$", lines[j]):
            return start, j
    raise SystemExit(f"entry {ENTRY_ID} has no config: block")


def strip_existing(lines: list[str], start: int, end: int) -> list[str]:
    """Drop any previously inserted comfyui* keys from the entry."""
    keys = ("comfyuiBaseURL:", "comfyuiActiveWorkflow:", "comfyuiTimeoutMs:", "comfyuiWorkflows:")
    out, i = [], start
    while i < end:
        line = lines[i]
        if any(line.startswith("    " + k) for k in keys):
            # Skip the key and any deeper-indented continuation lines.
            base = len(line) - len(line.lstrip())
            i += 1
            while i < end:
                nxt = lines[i]
                if not nxt.strip():
                    i += 1
                    continue
                if len(nxt) - len(nxt.lstrip()) > base:
                    i += 1
                    continue
                break
            continue
        out.append(line)
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not PATCH.exists():
        print(f"not found: {PATCH}", file=sys.stderr)
        return 2
    if not WORKFLOW.exists():
        print(f"not found: {WORKFLOW}", file=sys.stderr)
        return 2

    workflow_json = WORKFLOW.read_text(encoding="utf-8")
    json.loads(workflow_json)  # fail early if the workflow is not valid JSON

    original = PATCH.read_text(encoding="utf-8")
    lines = original.splitlines()

    start, config_idx = find_entry(lines)
    end = len(lines)
    for k in range(start + 1, len(lines)):
        if re.match(r"^-\s+", lines[k]):
            end = k
            break

    kept = strip_existing(lines, start, end)
    block = build_block(workflow_json).splitlines()

    # Re-locate config: after stripping, indices shift; find again in `kept`.
    _, config_idx = find_entry(kept)
    new_lines = kept[: config_idx + 1] + block + kept[config_idx + 1 :]
    updated = "\n".join(new_lines) + "\n"

    if args.check:
        print("would update:", PATCH)
        print(f"  entry      : {ENTRY_ID}")
        print(f"  workflow   : {WORKFLOW_NAME}")
        print(f"  base URL   : {BASE_URL}")
        print(f"  timeout    : {TIMEOUT_MS} ms")
        print(f"  json bytes : {len(workflow_json)}")
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = PATCH.with_suffix(PATCH.suffix + f".bak-{stamp}")
    shutil.copy2(PATCH, backup)
    print("backup:", backup)

    PATCH.write_text(updated, encoding="utf-8")
    print("wrote :", PATCH)

    # Validate the result parses and carries the expected settings.
    if yaml is None:
        # 没有 pyyaml 时退化为文本级自检：关键键必须都在，且只出现一次。
        text = PATCH.read_text(encoding="utf-8")
        must_have = ("comfyuiBaseURL:", "comfyuiActiveWorkflow:",
                     "comfyuiTimeoutMs:", "comfyuiWorkflows:")
        missing = [k for k in must_have if k not in text]
        dup = [k for k in must_have if text.count("\n    " + k) > 1]
        if missing or dup:
            print(f"VALIDATION WEAK: missing={missing} duplicated={dup}", file=sys.stderr)
            print("  （装个 pyyaml 才能做完整校验：pip install pyyaml）", file=sys.stderr)
            return 1
        print("\nvalidated (text-level, pyyaml 未安装):")
        print("  comfyui 键齐备且无重复；未做 YAML 结构解析")
        return 0

    parsed = yaml.safe_load(PATCH.read_text(encoding="utf-8"))
    entry = next((e for e in parsed if e.get("id") == ENTRY_ID), None)
    if entry is None:
        print("VALIDATION FAILED: entry missing after write", file=sys.stderr)
        return 1
    cfg = entry.get("config") or {}
    wfs = cfg.get("comfyuiWorkflows") or []
    print("\nvalidated:")
    print("  provider          :", cfg.get("provider"))
    print("  activeWorkflow    :", cfg.get("comfyuiActiveWorkflow"))
    print("  timeoutMs         :", cfg.get("comfyuiTimeoutMs"))
    print("  workflow count    :", len(wfs))
    for w in wfs:
        body = w.get("json") or ""
        ok = "{{prompt}}" in body and "{{image}}" in body
        print(f"  - {w.get('name')}  json={len(body)} bytes  placeholders={'OK' if ok else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
