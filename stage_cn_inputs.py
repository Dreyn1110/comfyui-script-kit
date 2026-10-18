"""Extract the sample input images the bundled workflows reference, and register them with
ComfyUI through /upload/image (the sanctioned path - the README rule about not copying files
into the install dir is about OUTPUT images; inputs must live in ComfyUI's input/ folder for
LoadImage to see them, and the upload endpoint puts them there for us).

    python comfyui/stage_cn_inputs.py          # extract + upload the two ControlNet samples
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsh  # noqa: E402
import paths  # noqa: E402

ZIP = paths.WORKFLOWS_DIR / "入门工作流" / "ComfyUI_Workflows.zip"
CACHE = dsh.HERE / "sample_input"
WANT = ["input/cwf_portrait.jpg", "input/cwf_dance.jpg"]


def main() -> int:
    if not ZIP.is_file():
        print(f"找不到 {ZIP}")
        return 1
    CACHE.mkdir(exist_ok=True)
    dsh.ensure_server(autostart=True)
    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        for n in WANT:
            if n not in names:
                print(f"[跳过] zip 里没有 {n}")
                continue
            dest = CACHE / Path(n).name
            if not dest.exists() or dest.stat().st_size == 0:
                dest.write_bytes(z.read(n))
            stored = dsh.upload_image(str(dest))
            print(f"[OK] {Path(n).name}  ({dest.stat().st_size // 1024} KB)  -> ComfyUI input: {stored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
