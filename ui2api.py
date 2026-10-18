"""Convert a ComfyUI **UI-format** workflow json (saved from the web canvas) into the
**API format** that POST /prompt accepts, optionally submit it and fetch the image.

Why this file exists
--------------------
Files saved from the web canvas look like {"nodes":[...], "links":[...]} and carry
positions, a link table and *positional* widget arrays. /prompt only eats
{"<id>": {"class_type": ..., "inputs": {...}}} with *named* inputs and links written as
["<src_id>", <slot>]. Nothing outside the browser converts them, so the 53 saved
workflows are unusable by any script until something does it.

The traps this handles (each one silently corrupts parameters instead of erroring):

1. **widgets_values has no names.** They are matched by position against the node type's
   widget inputs from /object_info. But the frontend injects a *control_after_generate*
   dropdown next to every seeded field (KSampler seed / noise_seed) that does NOT exist in
   /object_info, and *removes* a widget from widgets_values when it has been promoted to a
   wire. So the raw array can be longer or shorter than the definition list. Both are
   corrected before zipping; a leftover mismatch is reported instead of guessed around.
   Newer UI format (inputs entries carrying {"widget": {"name","value"}}) is read by name
   and needs no guessing.
2. **Combo values hold the author's paths.** Local checkpoints live in subfolders
   (SD1.5\\v1-5-pruned-emaonly.safetensors), so "v1-5-pruned-emaonly.safetensors" fails with
   value_not_in_list. Every combo value is validated against the live /object_info list and
   re-matched by filename.
3. **lite-graph helper nodes are not compute nodes.** Note / MarkdownNote / Separator are
   just dropped. Reroute carries a wire through itself, so links are resolved *through* the
   chain instead of pointing at a node we deleted. PrimitiveNode holds a constant that gets
   inlined into whatever slot consumed it.

Usage
-----
    python comfyui/ui2api.py 某工作流.json                 # convert, print API json only
    python comfyui/ui2api.py 某工作流.json --run           # convert + submit + save image
    python comfyui/ui2api.py 某工作流.json --run -o my.png
    python comfyui/ui2api.py 某工作流.json --set prompt="..." --set negative="..."
    python comfyui/ui2api.py 某工作流.json --set ckpt_name=Counterfeit-V3.0_fix_fp16.safetensors
    python comfyui/ui2api.py 某工作流.json --run --dry     # print the payload, do not submit
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsh  # noqa: E402  reuse ping / ensure_server / submit_and_wait / fetch_image

# One-shot, before anything can write: this file's own sys.exit messages go to stderr, and
# a GBK console turns them into mojibake. dsh.utf8_console() fixes both streams.
dsh.utf8_console()

API = dsh.API

# Compute nothing; pure canvas decoration.
DROP_NODES = {"Note", "MarkdownNote", "Separator", "Markdown"}
# Pass a wire through themselves; must be resolved, not emitted.
PASSTHROUGH_NODES = {"Reroute"}
# Hold a constant used by other nodes' widget inputs; inline the value.
CONSTANT_NODES = {"PrimitiveNode", "Anything", "Constant"}

# values that only ever come from the frontend's control_after_generate dropdown
CONTROL_TOKENS = {"randomize", "fixed", "increment", "decrement"}
# widgets the frontend decorates with that dropdown, i.e. widgets_values carries an extra
# entry right after each of them (verified on basic_parametrized.json: 7 values / 6 widgets)
SEEDISH = {"seed", "noise_seed"}
# lite-graph bookkeeping slots that carry no data.
NON_DATA_SLOTS = {"flow_control", "select", "title", "color", "bgcolor", "forcesize",
                  "nodesep", "ranksep", "bottom", "left", "top", "right", "control_after_generate"}
# node.mode values (litegraph): 0 = normal; the others mean the author disabled the node.
NODE_MODES = {0: "normal", 1: "bypass", 2: "never", 3: "on trigger", 4: "mute"}


# ------------------------------------------------------------------ /object_info
_cache: dict[str, dict | None] = {}


class ServerGone(RuntimeError):
    """The ComfyUI server stopped answering; conversion results are not trustworthy."""


def object_info(node_type: str):
    """Node definition from /object_info, or None if that type is not registered.

    Raises ServerGone when the endpoint itself is unreachable. A running server is required
    in the first place: widget names come from here, so without it every parameter of every
    node would be silently dropped and the payload would be worse than useless.
    """
    if node_type in _cache:
        return _cache[node_type]
    url = f"{API}/object_info/{urllib.parse.quote(node_type)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            info = json.load(r).get(node_type)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise ServerGone(f"ComfyUI 在转换过程中不可访问（{url}）: {e}") from None
    _cache[node_type] = info
    return info


_all_types: list[str] | None = None


def all_node_types() -> list[str]:
    """Every registered node class name, from one full /object_info call (cached)."""
    global _all_types
    if _all_types is None:
        try:
            with urllib.request.urlopen(f"{API}/object_info", timeout=180) as r:
                _all_types = sorted(json.load(r))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise ServerGone(f"取全量节点列表失败: {e}") from None
    return _all_types


def resolve_renamed(node_type: str) -> str | None:
    """Map a deprecated node name onto its current one.

    ComfyUI renames nodes and keeps the old id out of /object_info; the browser fixes those
    up through api.node_replacement (see comfy_extras/nodes_replacements.py), a script does
    not get that for free. Real case found here: old workflows store "ConditioningAverage "
    with a trailing space, which is now "ConditioningAverage". Matching on a normalised name
    (whitespace stripped, case folded) covers that whole family of renames without
    hard-coding the table, and stays correct as upstream adds more entries.
    """
    key = node_type.strip().lower()
    hits = [t for t in all_node_types() if t.strip().lower() == key]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # two live types differing only by whitespace/case: prefer an exact-normalised hit
        return node_type.strip() if node_type.strip() in hits else None
    return None


def _input_map(info: dict, group: str) -> dict:
    d = (info.get("input", {}) or {}).get(group)
    return d if isinstance(d, dict) else {}


# scalar widget types in the classic /object_info shape. Anything else a node lists
# (MODEL, CLIP, LATENT, CONDITIONING, IMAGE, VAE, ...) is a wire, not a widget.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "INT_RANGE", "FLOAT_RANGE"}


def is_widget_spec(spec) -> bool:
    """Decide whether one /object_info input is a widget or a connection.

    Measured on v0.30.2: there is NO {"widget": {...}} marker at all - every input looks
    identical - so the only signal is the type itself:
        ['INT', {..}] / ['FLOAT', {..}] / ['STRING', {..}]   -> widget
        [['euler', 'dpmpp_2m', ...], {..}]                  -> combo widget
        ['MODEL', {..}] / ['LATENT', {..}]                  -> wire
    """
    if isinstance(spec, dict) and spec.get("widget"):  # newer builds do mark it
        return True
    if not (isinstance(spec, list) and spec):
        return False
    head = spec[0]
    if isinstance(head, list):
        return True  # combo
    return isinstance(head, str) and head.upper() in WIDGET_TYPES


def widget_order(info: dict) -> list[str]:
    """Widget-input names in the order the frontend created them (required, then optional).

    KSampler checks out against the saved array: required dict order is
    model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,denoise and
    filtering to widgets gives seed,steps,cfg,sampler_name,scheduler,denoise - exactly the
    widgets_values order once control_after_generate is removed.
    """
    out, seen = [], set()
    for group in ("required", "optional"):
        for name, spec in _input_map(info, group).items():
            if name not in seen and is_widget_spec(spec):
                out.append(name)
                seen.add(name)
    return out


def combo_lists(info: dict) -> dict[str, list[str]]:
    """{input_name: [allowed values]} for combo widgets.

    A combo is spec = [['a','b',...], {meta}] (classic) or {"type": [...]} (marked build).
    Taking spec[0] matters: the bare list also contains the metadata dict, which would
    otherwise be stringified into the allowed-values list.
    """
    res = {}
    for group in ("required", "optional"):
        for name, spec in _input_map(info, group).items():
            vals = None
            if isinstance(spec, list) and spec and isinstance(spec[0], list):
                vals = spec[0]
            elif isinstance(spec, dict) and isinstance(spec.get("type"), list):
                vals = spec["type"]
            if vals is not None:
                res[name] = [str(x) for x in vals]
    return res


# ------------------------------------------------------------------ combo repair
def fix_combo(name: str, value, allowed: list[str], notes: list, nid: str, ntype: str):
    if not isinstance(value, str) or not allowed:
        return value
    if value in allowed:
        return value
    low = {a.lower(): a for a in allowed}
    if value.lower() in low:
        notes.append(f"    ~ #{nid} {ntype}.{name}: 大小写 -> '{low[value.lower()]}'")
        return low[value.lower()]
    base = Path(value.replace("\\", "/")).name.lower()
    cands = [a for a in allowed if Path(a.replace("\\", "/")).name.lower() == base]
    if len(cands) == 1:
        notes.append(f"    ~ #{nid} {ntype}.{name}: '{value}' -> '{cands[0]}' (本机路径)")
        return cands[0]
    if cands:
        notes.append(f"    ! #{nid} {ntype}.{name}: '{value}' 同名候选 {cands}，保留原值")
        return value
    notes.append(f"    ! #{nid} {ntype}.{name}: '{value}' 不在本机可选列表 -> 提交会报 value_not_in_list")
    return value


# ------------------------------------------------------------------ alignment
def _try_map(names: list[str], vals: list, tail_ok: bool = False, prefix_ok: bool = False):
    """Zip widget names to values, skipping the frontend control_after_generate entries.

    Returns the mapping only when every value is consumed exactly - that exact fit is the
    evidence this name list is the right one. Two version-skew relaxations:
      tail_ok   - leftover *string* values at the end are frontend-only widgets (LoadImage's
                  upload button). Numbers are never forgiven: a misplaced numeric widget is
                  precisely the silent corruption this whole function exists to catch.
      prefix_ok - there are fewer values than names because the node gained widgets after the
                  workflow was saved; the missing tail takes server defaults (what the browser
                  does too). Only valid when all values were consumed by the leading names.
    """
    out, i = {}, 0
    for name in names:
        if i >= len(vals):
            break          # ran out of values; remaining names may take defaults
        out[name] = vals[i]
        i += 1
        if name in SEEDISH and i < len(vals) and isinstance(vals[i], str) and vals[i] in CONTROL_TOKENS:
            i += 1   # the dropdown value, not a server input
    if i == len(vals):
        if len(out) == len(names) or prefix_ok:
            return out
        return None
    if tail_ok and i < len(vals) and all(isinstance(v, str) for v in vals[i:]):
        return out
    return None


def align(order: list[str], vals: list, wired: set, nid: str, ntype: str, notes: list) -> dict:
    """Pair widget names with values.

    Several serialisations coexist in saved files and they differ in ways that silently shift
    parameters if guessed wrong (steps landing in cfg, etc). Tried in order, each validated by
    "did every value get consumed":
      1. exact match against the full widget list.
      2. match against the not-yet-wired widgets - old builds drop a widget's value once it is
         promoted to a wire, whereas basic_parametrized.json keeps it (KSampler stored 7 values
         for 6 widgets). Both shapes occur in this machine's files.
      3. trailing string values are frontend-only widgets (LoadImage stores
         ["cwf_portrait.jpg", "image"]; the second is the upload button).
      4. fewer values than widgets because the node was upgraded after saving
         (DWPreprocessor stored 4, current definition has 7) - the tail takes server defaults.
    Only if all four fail is this a genuine anomaly worth the !! warning.
    """
    if not vals:
        return {}
    if not order:
        notes.append(f"    !! #{nid} {ntype}: 有 {len(vals)} 个 widget 值但识别不出输入名，值已丢弃")
        return {}
    mapped = _try_map(order, vals)
    if mapped is not None:
        return mapped
    unwired = [n for n in order if n not in wired]
    mapped = _try_map(unwired, vals)
    if mapped is not None:
        notes.append(f"    ok #{nid} {ntype}: 已转线的 widget 未存值（旧序列化），按未连线列表对齐")
        return mapped
    mapped = _try_map(order, vals, tail_ok=True)
    if mapped is not None:
        notes.append(f"    ok #{nid} {ntype}: 尾部 {len(vals) - len(order)} 个值是前端专属 widget（如上传按钮），已忽略")
        return mapped
    mapped = _try_map(order, vals, prefix_ok=True)
    if mapped is not None:
        notes.append(f"    ok #{nid} {ntype}: 旧文件只存了 {len(vals)} 个值，尾部 {len(order) - len(vals)} 个 widget 用服务端默认值")
        return mapped
    notes.append(
        f"    !! #{nid} {ntype}: widget 值 {len(vals)} 个 / 定义 {len(order)} 个（已连线 {len(wired)}），"
        f"四种对齐都不吻合，按位置硬配 -> {json.dumps(vals, ensure_ascii=False)[:150]}"
    )
    return dict(zip(order, vals))


# ------------------------------------------------------------------ graph walk
def link_index(ui: dict) -> dict:
    idx = {}
    for l in ui.get("links") or []:
        if isinstance(l, list) and len(l) >= 5:
            idx[l[0]] = {"src": str(l[1]), "src_slot": l[2], "dst": str(l[3]), "dst_slot": l[4]}
    return idx


def node_by_id(ui: dict) -> dict:
    return {str(n.get("id")): n for n in ui.get("nodes") or []}


def resolve_origin(nid: str, slot: int, nodes: dict, links: dict, notes: list, seen=None):
    """Follow a node output backwards through Reroute chains to a real node output.

    Returns [node_id, output_slot], or None when the chain ends in a constant node (the
    caller inlines its value) or is broken. flow_control-style bookkeeping slots are skipped:
    following them walks into the execution-order chain instead of upstream data.
    """
    seen = seen or set()
    node = nodes.get(nid)
    if node is None:
        return None
    if node.get("type") not in PASSTHROUGH_NODES:
        return [nid, slot]
    if nid in seen:
        notes.append(f"    ! Reroute 环 #{nid}")
        return None
    seen = seen | {nid}
    for inp in node.get("inputs") or []:
        if not isinstance(inp, dict) or inp.get("name") in NON_DATA_SLOTS:
            continue
        lid = inp.get("link")
        if lid in links:
            lk = links[lid]
            return resolve_origin(lk["src"], lk["src_slot"], nodes, links, notes, seen)
    notes.append(f"    ! Reroute #{nid} 上游没有连线")
    return None


def constant_value(node: dict, out_slot: int):
    """The scalar a PrimitiveNode/Constant would emit.

    PrimitiveNode stores its value in widgets_values[0]; the trailing "fixed" is the same
    control_after_generate decoration as on real nodes, so index 0 is the value.
    """
    wv = node.get("widgets_values") or []
    for v in wv:
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
    return None


# ------------------------------------------------------------------ conversion
def convert(ui: dict, notes: list) -> dict:
    links, nodes = link_index(ui), node_by_id(ui)
    api: dict[str, dict] = {}
    for node in ui.get("nodes") or []:
        nid, ntype = str(node.get("id")), str(node.get("type", "?"))
        mode = node.get("mode", 0)
        if ntype in DROP_NODES:
            notes.append(f"    - 丢弃注释节点 #{nid} {ntype}")
            continue
        if ntype in PASSTHROUGH_NODES:
            continue          # resolved inline by resolve_origin; no log noise per node
        if ntype in CONSTANT_NODES:
            continue          # value inlined wherever a wire pointed at it
        # resolve deprecated/renamed ids before anything else keys off the type
        if object_info(ntype) is None:
            fixed = resolve_renamed(ntype)
            if fixed and fixed != ntype:
                notes.append(f"    ~ #{nid} '{ntype}' 是旧节点名，已映射为 '{fixed}'")
                ntype = fixed
        if mode in (2, 4):    # never / mute: API format cannot express it, so it is dropped
            notes.append(f"    !! #{nid} {ntype} 在画布上是 {NODE_MODES[mode]} 状态，API 无此概念，已跳过")
            continue
        if mode == 1:
            notes.append(f"    !! #{nid} {ntype} 是 bypass 模式，按普通节点转换（bypass 无法在 API 表达）")

        info = object_info(ntype)
        if info is None:
            notes.append(f"    !! #{nid} {ntype}: /object_info 查无此节点（未安装？）")
        inputs: dict[str, object] = {}
        # Three different notions, keeping them apart is what makes alignment work:
        #   promoted : slot carries a widget marker (may or may not be connected)
        #   linked   : slot actually has a data wire -> its value comes from upstream
        #   assigned : we resolved a value for it, wire or inlined constant
        promoted = {s["name"] for s in (node.get("inputs") or [])
                    if isinstance(s, dict) and isinstance(s.get("widget"), dict)}
        linked, assigned = set(), set()

        # 1) real data connections -> ["src_node", src_slot], or an inlined constant.
        #    inputs[].widget carries name+config but NO value, so nothing is read from it here.
        for slot in node.get("inputs") or []:
            if not isinstance(slot, dict) or slot.get("link") in (None, -1):
                continue
            name = slot.get("name")
            if name in NON_DATA_SLOTS:
                continue
            lk = links.get(slot["link"])
            if lk is None:
                continue
            linked.add(name)
            origin = resolve_origin(lk["src"], lk["src_slot"], nodes, links, notes)
            src = nodes.get(lk["src"])
            # A wire from a PrimitiveNode/Constant is a *value*, not a node reference: those
            # nodes are not real compute nodes and never appear in the API payload, so the
            # value must be inlined. resolve_origin returns them as [id, slot] unchanged,
            # which is why this is checked on the raw source as well as on origin.
            src_is_const = src is not None and src.get("type") in CONSTANT_NODES
            origin_is_const = (origin is not None and origin[0] in nodes
                               and nodes[origin[0]].get("type") in CONSTANT_NODES)
            if src_is_const or origin_is_const:
                cnode, cslot = (src, lk["src_slot"]) if src_is_const else (nodes[origin[0]], origin[1])
                val = constant_value(cnode, cslot)
                if val is None:
                    notes.append(f"    ! #{nid} {ntype}.{name} 常量节点 #{cnode.get('id')} 取不到值")
                    continue
                if info and name in combo_lists(info) and isinstance(val, str):
                    val = fix_combo(name, val, combo_lists(info)[name], notes, nid, ntype)
            elif origin is None:
                continue
            elif origin[0] not in nodes:
                notes.append(f"    ! #{nid} {ntype}.{name} 上游 #{origin[0]} 不存在，已断开")
                continue
            else:
                val = origin
            inputs[name] = val
            assigned.add(name)

        # 2) remaining widgets -> named inputs
        order = widget_order(info) if info else []
        mapped = align(order, node.get("widgets_values") or [], linked, nid, ntype, notes)
        combos = combo_lists(info) if info else {}
        for name, val in mapped.items():
            if name in assigned:
                continue      # a wire already supplied it
            if name in combos and isinstance(val, str):
                val = fix_combo(name, val, combos[name], notes, nid, ntype)
            inputs[name] = val
        if promoted - linked - assigned:
            notes.append(f"    ~ #{nid} {ntype}: {sorted(promoted - linked - assigned)} 转成了输入槽但没连线，用了保存的旧值")

        # never emit a null input: the server rejects the whole prompt for one
        for k in [k for k, v in inputs.items() if v is None]:
            del inputs[k]
        api[nid] = {"class_type": ntype, "_meta": {"title": node.get("title") or ntype}, "inputs": inputs}

    # required-input check: catch a dropped wire before the server rejects the whole prompt
    for nid, node in api.items():
        info = object_info(node["class_type"])
        if not info:
            continue
        for name, spec in _input_map(info, "required").items():
            if name in node["inputs"]:
                continue
            if not is_widget_spec(spec):
                notes.append(f"    !! #{nid} {node['class_type']}: 必需连线 '{name}' 缺失 -> 提交会报 bad_parameters")
            else:
                notes.append(f"    !! #{nid} {node['class_type']}: 必需参数 '{name}' 没取到值")
        for name, v in list(node["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and str(v[0]) not in api:
                notes.append(f"    !! #{nid} {node['class_type']}.{name} 指向已删除的 #{v[0]}，已移除")
                del node["inputs"][name]
    return api


# ------------------------------------------------------------------ --set
def classify_conditioners(api: dict):
    pos, neg = [], []
    for nid, node in api.items():
        if node["class_type"] != "CLIPTextEncode":
            continue
        for other in api.values():
            for k, v in other["inputs"].items():
                if v == [nid, 0]:
                    if k == "positive":
                        pos.append(nid)
                    elif k == "negative":
                        neg.append(nid)
    return list(dict.fromkeys(pos)), list(dict.fromkeys(neg))


def coerce(raw: str):
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lstrip("-").isdigit():
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


# names people type for "the prompt", which on CLIPTextEncode is actually called `text`
PROMPT_ALIASES = {"prompt": "pos", "positive": "pos", "positive_prompt": "pos",
                  "negative": "neg", "negative_prompt": "neg"}


def apply_sets(api: dict, pairs: list[str], notes: list | None = None) -> None:
    notes = notes if notes is not None else []
    for kv in pairs:
        if "=" not in kv:
            sys.exit(f"--set 需要 name=value，收到: {kv}")
        name, raw = kv.split("=", 1)
        val = coerce(raw)
        hit = 0
        for nid, node in api.items():
            if name not in node["inputs"]:
                continue
            cur = node["inputs"][name]
            # Never overwrite a wire with a scalar. `positive`/`negative` are real KSampler
            # input names holding ["6",0]-style links, so a naive name match there would
            # replace a connection with a string and corrupt the graph while still reporting
            # "hit". Only a list value may replace a list.
            if isinstance(cur, list) and not isinstance(val, list):
                continue
            # Drop convert()'s warning about the value we are about to replace, THEN validate
            # the new one - fix_combo emits the same "#<id> <type>.<name>: ... 不在本机可选列表"
            # shape, so clearing after it would erase a legitimate warning about the new value.
            stale = f".{name}:"
            notes[:] = [x for x in notes
                        if not (f"#{nid} " in x and stale in x and "不在本机可选列表" in x)]
            # The value the user typed may itself be an author-machine path; validate it
            # against the live combo list exactly as convert() does.
            info = object_info(node["class_type"])
            combos = combo_lists(info) if info else {}
            if isinstance(val, str) and name in combos:
                val = fix_combo(name, val, combos[name], notes, nid, node["class_type"])
            node["inputs"][name] = val
            hit += 1
        if not hit and name in PROMPT_ALIASES:
            pos, neg = classify_conditioners(api)
            tgt = pos if PROMPT_ALIASES[name] == "pos" else neg
            for nid in tgt:
                api[nid]["inputs"]["text"] = val
                hit += 1
            if not hit:
                print(f"    --set {name}=... 找不到可识别的{'正向' if name in ('prompt','positive','positive_prompt') else '负向'}"
                      f" CLIPTextEncode（中间隔了 ConditioningConcat 之类），请直接用 --set text=值")
        print(f"    --set {name}={json.dumps(val, ensure_ascii=False)[:70]} -> 命中 {hit} 处"
              + ("" if hit else "   << 没有这个输入名"))


# ------------------------------------------------------------------ cli
def main() -> None:
    ap = argparse.ArgumentParser(description="ComfyUI UI 格式 -> API 格式（可提交出图）")
    ap.add_argument("json", help="UI 格式工作流 json（网页保存/导出的那种）")
    ap.add_argument("--run", action="store_true", help="转换后提交并取回图片")
    ap.add_argument("--dry", action="store_true", help="只打印，不提交")
    ap.add_argument("-o", "--out", default=None, help="输出图片路径")
    ap.add_argument("--save", default=None, help="转换结果另存为 API 格式 json")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--timeout", type=int, default=400)
    ap.add_argument("--no-autostart", action="store_true")
    args = ap.parse_args()

    try:
        src = dsh.find_workflow(args.json)
    except dsh.AmbiguousWorkflow as e:
        sys.exit(str(e))
    if src is None:
        sys.exit(f"找不到工作流 json: {args.json}\n（找过 comfyui/、工作区根、{dsh.WF_DIR}；也可给绝对路径）")
    try:
        ui = json.loads(src.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"找不到文件: {args.json}")
    if not isinstance(ui, dict) or "nodes" not in ui:
        sys.exit(f"{src} 顶层没有 nodes —— 它已经是 API 格式，直接跑： dsh.py sd --json {src.name}")

    # /object_info is not optional here: widget names come from it, and without them every
    # parameter of every node would be silently dropped.
    if not dsh.ping():
        if args.no_autostart:
            sys.exit("ComfyUI 没在跑，而转换必须有 /object_info。启动：powershell -ExecutionPolicy Bypass -File comfyui\\start_comfyui.ps1")
        dsh.ensure_server(autostart=True)

    notes: list[str] = []
    try:
        api = convert(ui, notes)
        if args.sets:
            apply_sets(api, args.sets, notes)
    except ServerGone as e:
        # the server died between ping and here; anything printed would be garbage
        sys.exit(f"转换中断：{e}\n（服务中途退出，本机 16GB 内存下这是已知症状；重启后重试）")

    unknown = sorted({n["class_type"] for n in api.values()} - {k for k in _cache if _cache[k]})
    print(f"源   : {src}")
    print(f"节点 : {len(ui.get('nodes') or [])} 个 UI 节点 -> {len(api)} 个 API 节点  (已按 /object_info 校验)")
    if unknown:
        print(f"未注册节点类型: {', '.join(unknown)}")
    if notes:
        print("备注 :")
        print("\n".join(notes))
    if args.save:
        Path(args.save).write_text(json.dumps(api, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已存 : {args.save}")
    print(json.dumps(api, ensure_ascii=False, indent=2))

    if not args.run:
        print("\n加 --run 才会真正提交出图。")
        return
    if args.dry:
        print("\n--dry：不提交。")
        return
    if unknown:
        print("!! 有未注册节点类型，提交大概率被拒；先装对应自定义节点，或去掉那些节点。")

    imgs = dsh.submit_and_wait(api, timeout=args.timeout, label="ui2api")
    save_images(imgs, args.out)


def save_images(imgs: list, out_arg=None) -> list:
    """Fetch every output image into the workspace.

    A workflow may SaveImage several times and also emit PreviewImage frames, so -o has to
    stay meaningful for batches: a single output uses it verbatim, several get _1/_2 before
    the extension instead of silently ignoring what the user asked for.
    """
    dests = []
    base = None
    if out_arg:
        base = Path(out_arg) if Path(out_arg).is_absolute() else dsh.WS / out_arg
    for i, ref in enumerate(imgs):
        if base is None:
            dest = dsh.WS / f"ui2api_{ref['filename']}"
        elif len(imgs) == 1:
            dest = base
        else:
            stem, ext = str(base).rsplit(".", 1)
            dest = Path(f"{stem}_{i + 1}.{ext}")
        p, n = dsh.fetch_image(ref, dest)
        kind = "" if ref.get("type", "output") == "output" else "  [预览/中间图]"
        print(f"  -> {p}  ({n / 1024:.0f} KB){kind}")
        dests.append(p)
    return dests


if __name__ == "__main__":
    main()
