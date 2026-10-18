"""Batch-download the missing weights that make already-installed nodes usable.

Every destination path below was read out of the node's own source, not guessed:

  ipadapter/       ComfyUI_IPAdapter_plus/IPAdapterPlus.py:37  -> models/ipadapter
  clip_vision/     IPAdapter_plus/utils.py:13-27 get_clipvision_file() matches
                   r'(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model)' so the
                   h94 image_encoder is saved under its canonical ViT-H name.
  controlnet/      built-in ControlNetLoader -> models/controlnet
  RMBG/RMBG-2.0/   ComfyUI-RMBG/py/AILab_RMBG.py:94 + AVAILABLE_MODELS cache_dir
  sams/            Impact-Pack impact_pack.py:51 add_folder_path_and_extensions("sams")
  controlnet_aux/ckpts/<repo>/<file>
                   controlnet_aux util.py:37 annotator_ckpts_path = <pkg>/ckpts
                   and custom_hf_download local_dir = ckpts/<repo_id>/<filename>

Reuses dl_model.fetch(), so resume + retry + safetensors-header check all apply.
Idempotent: existing files are skipped, safe to re-run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dl_model  # noqa: E402

utf8 = getattr(dl_model, "utf8_console", None)
if utf8:
    utf8()

import paths

M = paths.MODELS_DIR
AUX = paths.COMFY_ROOT / "custom_nodes" / "comfyui_controlnet_aux" / "ckpts"
HF = "https://hf-mirror.com/"

# (repo/remote, dest Path, size hint MB)
JOBS = [
    ("h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors",
     M / "ipadapter" / "ip-adapter-plus_sd15.safetensors", 94),
    ("h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors",
     M / "ipadapter" / "ip-adapter-plus-face_sd15.safetensors", 94),
    # IPAdapter SD1.5 image encoder (CLIP-ViT-H-14). 2.4GB, the big one.
    ("h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
     M / "clip_vision" / "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors", 2411),
    # ControlNet tile, fp16 variant (lllyasviel only ships 1378MB fp32 .bin)
    ("comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors",
     M / "controlnet" / "control_v11f1e_sd15_tile_fp16.safetensors", 689),
    # RMBG-2.0 (BiRefNet) - node needs all four files in one dir
    ("1038lab/RMBG-2.0/resolve/main/model.safetensors",
     M / "RMBG" / "RMBG-2.0" / "model.safetensors", 844),
    ("1038lab/RMBG-2.0/resolve/main/config.json",
     M / "RMBG" / "RMBG-2.0" / "config.json", 0),
    ("1038lab/RMBG-2.0/resolve/main/birefnet.py",
     M / "RMBG" / "RMBG-2.0" / "birefnet.py", 0),
    ("1038lab/RMBG-2.0/resolve/main/BiRefNet_config.py",
     M / "RMBG" / "RMBG-2.0" / "BiRefNet_config.py", 0),
    # SAM ViT-B for Impact-Pack (its default filename, impact_server.py:25)
    ("ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec64.pth",
     M / "sams" / "sam_vit_b_01ec64.pth", 358),
    # DWPose: the two files the node's own defaults resolve to
    ("yzd-v/DWPose/resolve/main/yolox_l.onnx",
     AUX / "yzd-v/DWPose/yolox_l.onnx", 207),
    ("yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
     AUX / "yzd-v/DWPose/dw-ll_ucoco_384.onnx", 130),
    ("hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt",
     AUX / "hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt", 129),
]


def main() -> int:
    total = sum(j[2] for j in JOBS)
    print(f"共 {len(JOBS)} 个文件，约 {total/1024:.2f} GB\n")
    todo = []
    for remote, dest, size in JOBS:
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[跳过] {dest.name} 已存在 ({dl_model.human(dest.stat().st_size)})")
        else:
            todo.append((remote, dest, size))
    if not todo:
        print("全部就位。")
        return 0

    print(f"\n待下载 {len(todo)} 个：")
    for remote, dest, size in todo:
        print(f"   {size:>5} MB  {dest}")

    failed = []
    for i, (remote, dest, _size) in enumerate(todo, 1):
        url = HF + remote
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(todo)}] {remote}")
        # HEAD for the real Content-Length so fetch() can detect a truncated file.
        # The table's size hint is only for the summary line.
        try:
            _st, expect, _etag = dl_model.head(url)
        except Exception as e:
            print(f"   HEAD 失败，按无长度校验继续: {e}")
            expect = 0
        if not dl_model.fetch(url, dest, expect):
            failed.append(str(dest))
        time.sleep(0.5)

    print("\n" + "=" * 56)
    if failed:
        print(f"失败 {len(failed)} 个（重跑本脚本会续传）：")
        for f in failed:
            print("   ", f)
        return 1
    print("全部下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
