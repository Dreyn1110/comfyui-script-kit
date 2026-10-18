"""End-to-end proof that IPAdapter actually works now (static combo lists don't prove it).

Encodes the reference image through CLIP-ViT-H, loads ip-adapter-plus_sd15, and
applies it to a Counterfeit SD1.5 generation. If either weight is missing or in
the wrong folder this fails at runtime with a clear node error, which is exactly
what we want to catch here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dsh  # noqa: E402
import paths  # noqa: E402

REF = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.OUT_DIR / "太空大肥猫.png"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else paths.OUT_DIR / "IPAdapter测试.png"

dsh.ensure_server(True)
name = dsh.upload_image(str(REF))
print(f"[ipadapter] 参考图 -> {name}")

WF = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "Counterfeit-V3.0_fix_fp16.safetensors"}},
    "2": {"class_type": "IPAdapterModelLoader",
          "inputs": {"ipadapter_file": "ip-adapter-plus_sd15.safetensors"}},
    "3": {"class_type": "CLIPVisionLoader",
          "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"}},
    "4": {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["2", 0], "image": ["5", 0], "clip_vision": ["3", 0],
        "weight": 0.9, "weight_type": "linear", "combine_embeds": "concat",
        "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
    "5": {"class_type": "LoadImage", "inputs": {"image": name}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["1", 1], "text": "masterpiece, best quality, 1girl, red dress, standing in a field of sunflowers, anime illustration, soft sunlight"}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["1", 1], "text": "lowres, worst quality, bad anatomy, bad hands, extra fingers, deformed, watermark, text"}},
    "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
    "9": {"class_type": "KSampler", "inputs": {
        "seed": 31337, "steps": 26, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0]}},
    "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
    "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ipadapter_test", "images": ["10", 0]}},
}

imgs = dsh.submit_and_wait(WF, 420, "ipadapter")
dest, n = dsh.fetch_image(imgs[0], OUT)
print(f"→ {dest}  ({n} 字节)")
