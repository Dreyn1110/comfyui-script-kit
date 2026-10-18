# ComfyUI Script Kit

Scripts for driving a **local ComfyUI** install from the command line: text-to-image,
hires-fix, model upscaling, instruction-based image editing, and — the part that is
actually hard — **converting browser-saved workflows into something the API accepts**.

Built and tested on a 6 GB laptop GPU. Everything here works in that budget, or says
so explicitly.

---

## The problem this solves

You save a workflow from the ComfyUI web canvas. It looks like this:

```json
{"nodes": [...], "links": [...]}
```

You try to POST it to `/prompt`. It fails. That's because `/prompt` accepts a
**completely different shape**:

```json
{"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "..."}}}
```

Nothing in the browser converts between them. So every workflow you save is
**unusable from a script** until something translates it.

`ui2api.py` is that translator. It has to handle several traps that all fail
**silently** — a wrong parameter still produces a plausible-looking image, so you
don't notice the input landed in the wrong slot.

---

## Quick start

```bash
# 1. point at your install (once)
export COMFYUI_HOME=/path/to/ComfyUI       # Linux/macOS
$env:COMFYUI_HOME = "C:\ComfyUI"           # Windows PowerShell

# 2. see what you have
python dsh.py t2i --prompt "1girl, silver hair, moonlit garden"

# 3. edit an existing image with an instruction
python dsh.py edit --src photo.png --prompt "把纸张颜色换成淡黄色，字迹不变"

# 4. run a workflow you saved in the browser
python dsh.py wf --json my_workflow --dry     # convert only, print the API json
python dsh.py wf --json my_workflow -o out.png
```

If `COMFYUI_HOME` is unset, the scripts try a default install location and stop
with a clear, actionable error if it doesn't exist — rather than failing later with
an opaque `FileNotFoundError`.

---

## Commands

| Command | What it does | Typical time |
|---|---|---|
| `dsh.py t2i` | Anime text-to-image (Counterfeit SD1.5 + 2 LoRAs) | ~20 s |
| `dsh.py hires` | Same, plus latent 2× upscale + denoise pass | ~27 s |
| `dsh.py upscale --src X` | 4× model upscale of an existing image | ~15 s |
| `dsh.py edit --src X --prompt Y` | Qwen-Image-Edit instruction editing | 105–145 s |
| `dsh.py sd --json X` | Run a raw API-format workflow | varies |
| `dsh.py wf --json X` | Run a **browser-saved** workflow (auto-converts) | varies |

All accept `-o` for output path and `--no-autostart` to refuse starting the server.

---

## The conversion traps

These are the reason `ui2api.py` is 559 lines and not 50. Each one corrupts data
**without raising an error**.

### 1. `widgets_values` has no names

The array is positional, matched against the node type's widget inputs from
`/object_info`. But two things shift the positions:

- **The frontend injects a `control_after_generate` dropdown** next to every seeded
  field (`seed`, `noise_seed`). It does not exist in `/object_info`. So the array is
  one entry **longer** than the definition list.
- **A widget promoted to a wire disappears** from `widgets_values`. So the array is
  one entry **shorter**.

Both are corrected before zipping. A mismatch that can't be resolved is *reported*,
not silently guessed around.

### 2. Combo values are the author's local paths

Your checkpoint lives at `SD1.5\v1-5-pruned-emaonly.safetensors` but the workflow
says `v1-5-pruned-emaonly.safetensors`. Combo validation fails with
`value_not_in_list` — which at least *does* error, unlike the others.

Every combo value is validated against the live `/object_info` list and
re-matched by filename.

### 3. Helper nodes are not compute nodes

- `Note` / `MarkdownNote` / `Separator` are canvas decoration → dropped.
- `Reroute` carries a wire through itself → links are resolved **through** the
  chain, not pointed at a node that got deleted.
- `PrimitiveNode` holds a constant consumed by other nodes → the value is inlined
  into whatever slot used it.

Skipping any of these produces a workflow with dangling references.

### 4. `-1` is not `0`

Where the protocol allows a sensor to be absent, the sentinel is `-1`, never `0`.
`0` is a legal temperature. Treating "no sensor" as `0 °C` yields a conclusion of
"your room is a freezer" from data that simply didn't exist.

---

## Running the tests

```bash
# converter regression suite — needs ComfyUI running (uses /object_info)
python test_ui2api.py
```

The suite includes **negative cases**: a genuinely misaligned widget array must still
produce a warning. A converter that silently produces a plausible image from garbage
input is worse than one that crashes.

---

## 6 GB VRAM / 16 GB RAM: what actually matters

This is the configuration everything here was tuned for, and the failure mode is
not what you'd guess.

**The crash was RAM, not VRAM.** ComfyUI was killed by the OS mid-run with no
traceback. Measured commit memory showed a **39.8 GB peak with free physical RAM
hitting 0.0 GB** — the pagefile ballooned to 43 GB.

`start_comfyui.ps1` adds `--cache-none`, which drops the peak to **28.8 GB**:

| Setting | Peak commit | Min free RAM |
|---|---|---|
| default | **39.8 GB** | **0.0 GB** |
| `--cache-none` | **28.8 GB** | 0.3 GB |

### `--lowvram` is a no-op here

Do not set it. From ComfyUI's own `cli_args.py`:

> `--lowvram`: Doesn't do anything if dynamic vram is enabled.

Dynamic VRAM is enabled unless you pass `--disable-dynamic-vram / --highvram /
--gpu-only / --novram / --cpu`. It's on by default, so `--lowvram` is silently
ignored. It also targets the wrong resource: it moves the text encoder between CPU
and GPU, while the bottleneck was system memory.

### Do not use `--disable-all-custom-nodes` to fix a hung boot

It disables ComfyUI-GGUF too, and the Qwen-Image-Edit workflow needs
`UnetLoaderGGUF` / `CLIPLoaderGGUF`. The whole edit chain becomes unknown node
classes. To work around a hung ComfyUI-Manager self-check, rename **only** the
Manager directory — that's what `start_comfyui.ps1 -NoManager` does.

---

## Model download notes

`dl_model.py` pulls from `hf-mirror.com` with resume and a safetensors header check.
`download_models.py` fetches the Qwen-Image-Edit stack (~14 GB).

**One trap worth knowing**: `ComfyUI-GGUF` pairs the `mmproj` file with its text
encoder **by filename prefix**. Upstream ships `mmproj-BF16.gguf`, which does not
contain the encoder's prefix, so pairing fails and Qwen-Image-Edit breaks with:

```
Can't find mmproj file ... Qwen-Image-Edit will be broken!
```

`download_models.py` saves it as `Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf` so the
prefix matches. This is a filename coincidence the loader depends on, not a
documented API.

---

## Files

```
paths.py              env-var path resolution (COMFYUI_HOME / COMFYUI_API / COMFYUI_OUT)
dsh.py                main entry: t2i / hires / upscale / edit / sd / wf
ui2api.py             UI-format -> API-format converter (the hard part)
start_comfyui.ps1     launches ComfyUI with memory-sane flags for small GPUs
dl_model.py           generic hf-mirror downloader (resume + header verification)
download_models.py    the Qwen-Image-Edit model set
survey_workflows.py   inventory: what your saved workflows need vs what you have
install_workflow.py   registers a workflow into the DSH image-gen plugin config
test_ui2api.py        converter regression suite (includes negative cases)
test_workflow.py      end-to-end: upload -> submit -> fetch
test_ipadapter.py     proves IPAdapter weights actually load, not just appear in a list
fill_gaps.py          downloads weights that make already-installed nodes usable
stage_cn_inputs.py    extracts ControlNet sample inputs and registers them
verify_rmbg_tile.py   proves RMBG and ControlNet-tile really run
show_api.py           pretty-prints an API-format workflow as node -> input
```

---

## License

MIT — see `LICENSE`.

Model weights are **not** included. Several of them have their own licenses
(SD1.5 CreativeML OpenRAIL-M, Qwen-Image-Edit Apache-2.0); check upstream before
redistributing anything you generate.
