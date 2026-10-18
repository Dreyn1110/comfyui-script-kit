"""Verify RMBG background removal and ControlNet tile end-to-end.

Both were 'installed but unusable' before the weights landed. A non-empty combo
list does not prove the file loads, so each gets a real /prompt run.
Field names/specs were read from live /object_info, not guessed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dsh  # noqa: E402
import paths  # noqa: E402
dsh.utf8_console()

WS = paths.OUT_DIR
SRC = WS / "太空大肥猫.png"  # 768x512, small enough for RMBG's process_res

dsh.ensure_server(True)
name = dsh.upload_image(str(SRC))
print(f"[verify] 输入图 {SRC.name} -> {name}")

# ---- 1. RMBG-2.0 cutout (Alpha = transparent output) ----
rmbg = {
    "2": {"class_type": "LoadImage", "inputs": {"image": name}},
    "1": {"class_type": "RMBG", "inputs": {"image": ["2", 0], "model": "RMBG-2.0",
          "sensitivity": 1.0, "process_res": 1024, "mask_blur": 0, "mask_offset": 0,
          "invert_output": False, "refine_foreground": False, "background": "Alpha"}},
    "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "rmbg_test", "images": ["1", 0]}},
}
try:
    imgs = dsh.submit_and_wait(rmbg, 300, "rmbg")
    dest, n = dsh.fetch_image(imgs[0], WS / "抠图测试.png")
    print(f"✅ RMBG-2.0 → {dest.name} ({n} 字节)")
except SystemExit as e:
    print("❌ RMBG 失败:", str(e)[:400])
except Exception as e:
    print("❌ RMBG 失败:", type(e).__name__, str(e)[:400])

dsh.free_vram()

# ---- 2. ControlNet tile: upscale source, VAEEncode, low-denoise detail pass ----
# EmptyLatentImage would be wrong here: tile is an img2img refiner, it must start
# from the encoded image or the result is a fresh unrelated picture.
tile = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "SD1.5\\v1-5-pruned-emaonly.safetensors"}},
    "2": {"class_type": "ControlNetLoader",
          "inputs": {"control_net_name": "control_v11f1e_sd15_tile_fp16.safetensors"}},
    "3": {"class_type": "LoadImage", "inputs": {"image": name}},
    "12": {"class_type": "ImageScale", "inputs": {
        "image": ["3", 0], "upscale_method": "lanczos", "width": 1024, "height": 688, "crop": "disabled"}},
    "4": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["1", 1], "text": "photograph, highly detailed, sharp focus, masterpiece"}},
    "5": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["1", 1], "text": "blurry, lowres, watermark, text, deformed"}},
    "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["12", 0], "vae": ["1", 2]}},
    "7": {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["4", 0], "negative": ["5", 0], "control_net": ["2", 0], "image": ["12", 0],
        "strength": 0.9, "start_percent": 0.0, "end_percent": 1.0}},
    "8": {"class_type": "KSampler", "inputs": {
        "seed": 5150, "steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.42, "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
        "latent_image": ["9", 0]}},
    "10": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
    "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "tile_test", "images": ["10", 0]}},
}
try:
    imgs = dsh.submit_and_wait(tile, 420, "tile")
    dest, n = dsh.fetch_image(imgs[0], WS / "Tile放大测试.png")
    print(f"✅ ControlNet tile → {dest.name} ({n} 字节)")
except SystemExit as e:
    print("❌ tile 失败:", str(e)[:400])
except Exception as e:
    print("❌ tile 失败:", type(e).__name__, str(e)[:400])
