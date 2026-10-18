"""Regression tests for ui2api.convert() - run after ANY change to the converter.

Why a real test file: every bug found in this converter was a SILENT one (a parameter landing
in the wrong input, a value dropped) that still produced a plausible image. The PrimitiveNode
override bug could not be caught with the bundled example workflows at all - their primitive
values happen to equal the stale widget values, so a broken converter passes. Hence the
synthetic cases below, including NEGATIVE ones (a genuinely misaligned array must still warn).

    python comfyui/test_ui2api.py         # needs ComfyUI running (uses /object_info)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsh  # noqa: E402
import ui2api  # noqa: E402

dsh.utf8_console()

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))


def convert_text(ui: dict):
    notes: list[str] = []
    api = ui2api.convert(ui, notes)
    return api, notes


def ksampler_ui(ks_widgets, primitives):
    """A KSampler whose widgets are overridden by PrimitiveNodes; primitives = {input: value}."""
    nodes = [
        {"id": 3, "type": "KSampler", "mode": 0, "widgets_values": ks_widgets,
         "inputs": [{"name": k, "type": "ANY", "link": 100 + i,
                     "widget": {"name": k, "config": ["X", {}]}} for i, k in enumerate(primitives)],
         "outputs": [{"name": "LATENT", "type": "LATENT", "links": [], "slot_index": 0}]},
    ]
    links = []
    for i, (k, v) in enumerate(primitives.items()):
        pid = 90 + i
        nodes.append({"id": pid, "type": "PrimitiveNode", "mode": 0, "widgets_values": [v, "fixed"],
                      "outputs": [{"name": "X", "type": "INT", "links": [100 + i], "slot_index": 0,
                                   "widget": {"name": k, "config": ["X", {}]}}]})
        links.append([100 + i, pid, 0, 3, i, "INT"])
    return {"nodes": nodes, "links": links}


def main() -> int:
    if not dsh.ping():
        print("需要 ComfyUI 在跑（转换器依赖 /object_info）")
        return 2

    print("== 1. 对齐规则 ==")
    def al(order, vals, wired=frozenset()):
        n = []
        return ui2api.align(order, vals, set(wired), "1", "X", n), n

    m, n = al(["image"], ["cwf.jpg", "image"])
    check("LoadImage 尾部字符串被忽略", m == {"image": "cwf.jpg"}, str(m))
    m, n = al(["w", "h"], [512, 512, 1])
    check("尾部数字仍报警（不许赦免）", any("!!" in x for x in n), str(n))
    m, n = al(["a", "b", "c"], [1, "x"])
    check("少值走默认不报警", not any("!!" in x for x in n) and m == {"a": 1, "b": "x"}, str(n))
    m, n = al(["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
              [6, "fixed", 20, 7, "dpmpp_2m", "karras", 1])
    check("control_after_generate 被丢弃且不错位",
          m == {"seed": 6, "steps": 20, "cfg": 7, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1}, str(m))

    print("== 2. PrimitiveNode 覆盖（真实示例值相同，测不出来，必须用不同值）==")
    api, _ = convert_text(ksampler_ui([111, "fixed", 20, 7, "euler", "karras", 1],
                                      {"steps": 33, "cfg": 3.5}))
    ks = api["3"]["inputs"]
    check("primitive 值覆盖 KSampler 残留值", ks["steps"] == 33 and ks["cfg"] == 3.5, str(ks))
    check("未覆盖的仍取自身值", ks["seed"] == 111 and ks["sampler_name"] == "euler", str(ks))
    check("payload 里没有 PrimitiveNode", all(v["class_type"] != "PrimitiveNode" for v in api.values()))

    print("== 3. Reroute 展平 / Note 丢弃 ==")
    api, _ = convert_text({
        "nodes": [
            {"id": 1, "type": "EmptyLatentImage", "mode": 0, "widgets_values": [512, 512, 1],
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [10], "slot_index": 0}]},
            {"id": 2, "type": "Reroute", "mode": 0, "inputs": [{"name": "", "link": 10}],
             "outputs": [{"name": "", "type": "*", "links": [11], "slot_index": 0}]},
            {"id": 3, "type": "Reroute", "mode": 0, "inputs": [{"name": "", "link": 11}],
             "outputs": [{"name": "", "type": "*", "links": [12], "slot_index": 0}]},
            {"id": 4, "type": "KSampler", "mode": 0, "widgets_values": [1, "fixed", 20, 8, "euler", "normal", 1],
             "inputs": [{"name": "latent_image", "type": "LATENT", "link": 12}]},
            {"id": 5, "type": "Note", "mode": 0},
        ],
        "links": [[10, 1, 0, 2, 0, "*"], [11, 2, 0, 3, 0, "*"], [12, 3, 0, 4, 0, "LATENT"]],
    })
    check("链式 Reroute 追到真正的源头", api["4"]["inputs"]["latent_image"] == ["1", 0], str(api["4"]["inputs"]))
    check("Note 被丢弃", "5" not in api)
    check("Reroute 不进 payload", all(v["class_type"] != "Reroute" for v in api.values()))

    print("== 4. 旧节点名改名 ==")
    api, notes = convert_text({"nodes": [{"id": 1, "type": "ConditioningAverage ", "mode": 0,
                                          "widgets_values": [0.5], "inputs": []}], "links": []})
    check("带尾空格的旧名映射到现行名", api["1"]["class_type"] == "ConditioningAverage", str(api["1"]))
    check("改名记进备注", any("旧节点名" in x for x in notes), str(notes))

    print("== 5. --set 不破坏连线 + 校验新值 ==")
    ui = json.loads(dsh.find_workflow("clip_skip").read_text(encoding="utf-8"))
    api, notes = convert_text(ui)
    ks_id = next(k for k, v in api.items() if v["class_type"] == "KSampler")
    neg_wire = api[ks_id]["inputs"]["negative"]
    ui2api.apply_sets(api, ["negative=MY_NEG"], notes)
    check("KSampler.negative 连线未被字符串覆盖",
          api[ks_id]["inputs"]["negative"] == neg_wire, str(api[ks_id]["inputs"]["negative"]))
    check("--set negative 命中真正的负向 CLIPTextEncode",
          any(v["class_type"] == "CLIPTextEncode" and v["inputs"].get("text") == "MY_NEG" for v in api.values()))

    ui = json.loads(dsh.find_workflow("canny").read_text(encoding="utf-8"))
    api, notes = convert_text(ui)
    check("转换时对缺失底模报警", any("不在本机" in x for x in notes), str(notes))
    ui2api.apply_sets(api, ["ckpt_name=SD1.5\\v1-5-pruned-emaonly.safetensors"], notes)
    check("覆盖成合法值后旧报警清除", not any("不在本机" in x for x in notes), str(notes))
    api, notes = convert_text(ui)
    ui2api.apply_sets(api, ["ckpt_name=nope.safetensors"], notes)
    check("覆盖成非法值时仍报警", any("nope.safetensors" in x and "不在本机" in x for x in notes), str(notes))

    print("== 6. 全部可跑工作流结构自检 ==")
    RUN = ["basic_workflow", "basic_latent_batch", "basic_parametrized", "basic_workflow_ext_vae",
           "blank_token", "clip_skip", "compare_samplers", "conditioning_average", "conditioning_concat",
           "denoise_impact_on_latent", "embeddings", "horrors_negative", "save_noise_steps",
           "upscale_by_model", "upscale_latent", "upscale_pixel", "word_weighting", "canny", "pose"]
    for name in RUN:
        p = dsh.find_workflow(name)
        if p is None:
            check(f"{name} 可定位", False)
            continue
        ui = json.loads(p.read_text(encoding="utf-8"))
        notes = []
        api = ui2api.convert(ui, notes)
        hard = [x for x in notes if "!!" in x]
        dangling = [(k, i) for k, v in api.items() for i, j in v["inputs"].items()
                    if isinstance(j, list) and str(j[0]) not in api]
        nulls = [(k, i) for k, v in api.items() for i, j in v["inputs"].items() if j is None]
        check(f"{name} 转换干净", not (hard or dangling or nulls),
              f"hard={hard} dangling={dangling} nulls={nulls}")

    print(f"\n通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败:", ", ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
