"""Run the Qwen-Image-Edit workflow against a live ComfyUI server, end to end.

This mirrors what the DSH image-gen plugin does when it calls ComfyUI:
  1. upload the source image to /upload/image
  2. substitute {{prompt}} / {{seed}} / {{image}} in the workflow JSON
  3. POST the workflow to /prompt
  4. poll /history/<prompt_id> until it finishes
  5. fetch the first output image from /view

Usage:
    python test_workflow.py <workflow.json> <source-image> "<prompt>" [out.png]

Exit code is non-zero if validation, execution, or image retrieval fails.
"""

from __future__ import annotations

import json
import mimetypes
import sys
import time
import urllib.parse
import urllib.request
import uuid

BASE = "http://127.0.0.1:8188"
TIMEOUT_S = 3600
POLL_S = 3.0


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.load(r)


def upload_image(path: str) -> str:
    """Upload one image and return the name ComfyUI stored it under."""
    boundary = "----dsh" + uuid.uuid4().hex
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        blob = fh.read()

    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
            f"Content-Type: {ctype}\r\n\r\n".encode(),
            blob,
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
            f"--{boundary}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        BASE + "/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        info = json.load(r)
    stored = info.get("name") or name
    sub = info.get("subfolder") or ""
    return f"{sub}/{stored}" if sub else stored


def prepare(workflow_path: str, prompt: str, image: str) -> dict:
    with open(workflow_path, encoding="utf-8") as fh:
        wf = json.load(fh)

    seed = int.from_bytes(uuid.uuid4().bytes[:4], "big")
    image_hits = 0

    def walk(value):
        nonlocal image_hits
        if isinstance(value, str):
            if value == "{{image}}":
                image_hits += 1
                return image
            if value == "{{seed}}":
                return seed
            return value.replace("{{prompt}}", prompt)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    prepared = {nid: walk(node) for nid, node in wf.items()}
    print(f"  seed={seed} image={image!r} image-placeholders={image_hits}")
    return prepared


def run(workflow: dict) -> dict:
    client_id = uuid.uuid4().hex
    try:
        res = post_json("/prompt", {"prompt": workflow, "client_id": client_id})
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"\nComfyUI rejected the workflow (HTTP {exc.code}):\n{detail}")
        raise

    prompt_id = res["prompt_id"]
    print(f"  queued prompt_id={prompt_id}")
    print("  number in queue:", res.get("number"))

    started = time.time()
    last_note = 0.0
    while True:
        hist = get_json(f"/history/{prompt_id}")
        if prompt_id in hist:
            return hist[prompt_id]
        if time.time() - started > TIMEOUT_S:
            raise TimeoutError(f"no result after {TIMEOUT_S}s")
        elapsed = time.time() - started
        if elapsed - last_note > 15:
            last_note = elapsed
            try:
                q = get_json("/queue")
                running = len(q.get("queue_running") or [])
                pending = len(q.get("queue_pending") or [])
                print(f"  ... {elapsed:5.0f}s  running={running} pending={pending}", flush=True)
            except Exception:
                print(f"  ... {elapsed:5.0f}s", flush=True)
        time.sleep(POLL_S)


def save_outputs(entry: dict, out_path: str) -> list[str]:
    saved = []
    for node_id, node_out in (entry.get("outputs") or {}).items():
        for img in node_out.get("images") or []:
            q = urllib.parse.urlencode(
                {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                }
            )
            with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=300) as r:
                data = r.read()
            target = out_path if not saved else out_path.rsplit(".", 1)[0] + f"_{len(saved)}." + out_path.rsplit(".", 1)[1]
            with open(target, "wb") as fh:
                fh.write(data)
            saved.append(target)
            print(f"  saved {target} ({len(data) / 1024:.0f} KB) from node {node_id}")
    return saved


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    workflow_path, image_path, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "shots/edit_result.png"

    print("[1/4] uploading source image")
    image_name = upload_image(image_path)
    print(f"  stored as {image_name!r}")

    print("[2/4] preparing workflow")
    workflow = prepare(workflow_path, prompt, image_name)

    print("[3/4] running")
    entry = run(workflow)

    status = (entry.get("status") or {}).get("status_str", "unknown")
    print(f"  status: {status}")
    if status != "success":
        print(json.dumps(entry.get("status"), indent=2)[:3000])
        return 1

    print("[4/4] fetching output")
    saved = save_outputs(entry, out_path)
    if not saved:
        print("  no images in output!")
        print(json.dumps(entry.get("outputs"), indent=2)[:2000])
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
