#!/usr/bin/env python3
"""
Smoke test for the GROBID trainer service.

Exercises the full model-versioning workflow:
  1. health check
  2. train with fast settings + keep_existing  →  new timestamped model file
  3. verify the file appears in GET /models/{model}
  4. evaluate (mode=1) using the timestamped model file
  5. delete the timestamped file
  6. verify the file is gone

Usage:
    python grobid-trainer/scripts/smoke_test.py [--host HOST] [--port PORT] [--model MODEL] [--flavor FLAVOR]

The test prints a summary and exits 0 on success, 1 on any failure.
Remove this file when it is no longer needed.
"""

import atexit
import argparse
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Cleanup registry — deletes the trained model file on abnormal exit
# ---------------------------------------------------------------------------

_cleanup: dict = {"base": None, "model": None, "trained_file": None, "flavor": ""}


def _cleanup_trained_file():
    if not _cleanup["trained_file"]:
        return
    base         = _cleanup["base"]
    model        = _cleanup["model"]
    trained_file = _cleanup["trained_file"]
    flavor       = _cleanup["flavor"]
    try:
        params = urllib.parse.urlencode({"name": trained_file, "flavor": flavor})
        req = urllib.request.Request(
            f"{base}/models/{model}?{params}", method="DELETE"
        )
        with urllib.request.urlopen(req):
            pass
        print(f"\n  [cleanup] deleted {trained_file}", file=sys.stderr)
    except Exception as exc:
        print(f"\n  [cleanup] could not delete {trained_file}: {exc}", file=sys.stderr)


atexit.register(_cleanup_trained_file)

# ---------------------------------------------------------------------------
# Re-use HTTP helpers from the CLI client
# ---------------------------------------------------------------------------

_CLIENT = Path(__file__).parent / "training-api-client.py"
spec = importlib.util.spec_from_file_location("training_api_client", _CLIENT)
_client_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_client_mod)


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        fail(f"HTTP {e.code} GET {url}\n    {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        fail(f"Cannot reach service at {url}: {e.reason}")


def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        fail(f"HTTP {e.code} POST {url}\n    {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        fail(f"Cannot reach service at {url}: {e.reason}")


def _delete(url):
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        fail(f"HTTP {e.code} DELETE {url}\n    {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        fail(f"Cannot reach service at {url}: {e.reason}")


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def step(label):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


def ok(msg=""):
    print(f"  ✓  {msg}" if msg else "  ✓")


def fail(msg):
    print(f"\n  ✗  FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def _sse_stream_compact(url: str) -> int:
    """Stream SSE, overwriting a single status line instead of scrolling."""
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(url, headers={"Accept": "text/event-stream"})
        )
    except urllib.error.HTTPError as e:
        fail(f"HTTP {e.code} streaming {url}\n    {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        fail(f"Cannot reach service at {url}: {e.reason}")

    cols = 76  # characters available for the log line
    buf = b""
    last_event = ""
    exit_code = 1
    try:
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.decode(errors="replace").rstrip("\r")
                if line.startswith("event:"):
                    last_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    text = line[len("data:"):].strip()
                    if last_event == "done":
                        print()   # leave the last status line visible
                        if text.startswith("exit_code="):
                            try:
                                exit_code = int(text.split("=", 1)[1])
                            except ValueError:
                                exit_code = 1
                        return exit_code
                    truncated = text[:cols].ljust(cols)
                    print(f"\r  {truncated}", end="", flush=True)
                elif line == "":
                    last_event = ""
    except KeyboardInterrupt:
        print()
        return 130
    print()
    return exit_code


def wait_for_job(base_url, job_id, *, poll_interval=5):
    """Stream job log via SSE (compact single-line display), then return the final job dict."""
    stream_url = f"{base_url}/jobs/{job_id}/stream"
    # Verify the job appears in the list before streaming starts.
    jobs = _get(f"{base_url}/jobs")
    if not any(j.get("job_id") == job_id for j in jobs):
        fail(f"job {job_id} not found in GET /jobs before streaming started")
    print(f"  [job {job_id}]")
    exit_code = _sse_stream_compact(stream_url)
    job = _get(f"{base_url}/jobs/{job_id}")
    if job["status"] != "done" or exit_code != 0:
        fail(
            f"job {job_id} ended with status={job['status']!r}, "
            f"exit_code={exit_code}\n"
            f"last log lines:\n"
            + "\n".join(("    " + l) for l in job.get("log", "").splitlines()[-20:])
        )
    return job


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smoke test for GROBID trainer service")
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   type=int, default=8072)
    parser.add_argument("--model",  default="date",
                        help="Model to use for the smoke test (default: date)")
    parser.add_argument("--flavor", default="",
                        help="Model flavor (default: none)")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    model  = args.model
    flavor = args.flavor
    flavor_suffix = f" (flavor={flavor})" if flavor else ""

    print(f"Smoke test  →  {base}  model={model}{flavor_suffix}")

    # ── 1. Health ────────────────────────────────────────────────────────────
    step("1/6  Health check")
    h = _get(f"{base}/health")
    if h.get("status") != "ok":
        fail(f"health status={h.get('status')!r}")
    if not h.get("jar_built"):
        fail("trainer JAR not built — run: ./gradlew :grobid-trainer:shadowJar --no-daemon")
    if model not in h.get("models", []):
        fail(f"model '{model}' not found in grobid-home/models/. "
             f"Available: {h.get('models')}")
    ok(f"platform={h['platform']}  jar={Path(h['jar']).name}")

    # ── 2. Train (fast, keep_existing) ───────────────────────────────────────
    step("2/6  Train with fast settings + keep_existing")
    payload = {
        "mode":             0,
        "keep_existing":    True,
        "epsilon":          0.001,
        "nb_max_iterations": 200,
    }
    if flavor:
        payload["flavor"] = flavor
    r = _post(f"{base}/train/{model}", payload)
    job_id = r["job_id"]
    ok(f"job started: {job_id}")

    train_job = wait_for_job(base, job_id)
    trained_file = train_job.get("trained_model_file")
    if not trained_file:
        fail("job succeeded but 'trained_model_file' is missing from job record")
    if trained_file == "model.wapiti":
        fail(
            "'trained_model_file' is 'model.wapiti' — keep_existing should have "
            "produced a timestamped filename. "
            "This often means the model directory was not where the service expected it. "
            f"Check that grobid-home/models/{model}{'/' + flavor if flavor else ''} exists."
        )
    ok(f"trained_model_file = {trained_file}")
    _cleanup.update({"base": base, "model": model, "trained_file": trained_file, "flavor": flavor})

    # Verify trained_model_file is visible in the job list (checked before eval SSE).
    jobs = _get(f"{base}/jobs")
    train_entry = next((j for j in jobs if j.get("job_id") == job_id), None)
    if train_entry is None:
        fail(f"training job {job_id} missing from GET /jobs after completion")
    list_tmf = train_entry.get("trained_model_file")
    if list_tmf != trained_file:
        fail(f"GET /jobs shows trained_model_file={list_tmf!r}, expected {trained_file!r}")
    ok(f"job list confirms trained_model_file = {trained_file}")

    # ── 3. Verify file appears in model listing ───────────────────────────────
    step("3/6  Verify new model file is listed")
    params = f"?flavor={urllib.parse.quote(flavor)}" if flavor else ""
    listing = _get(f"{base}/models/{model}{params}")
    names = [f["name"] for f in listing["files"]]
    if trained_file not in names:
        fail(f"expected '{trained_file}' in model listing, got: {names}")
    active_names = [f["name"] for f in listing["files"] if f["is_active"]]
    if "model.wapiti" not in active_names:
        fail(f"model.wapiti should still be active after keep_existing, got active={active_names}")
    ok(f"found {trained_file}  (active model is still model.wapiti)")

    # ── 4. Evaluate using the timestamped model file ──────────────────────────
    step("4/6  Evaluate using the timestamped model file (mode=1)")
    eval_payload = {
        "mode":       1,
        "model_file": trained_file,
    }
    if flavor:
        eval_payload["flavor"] = flavor
    r = _post(f"{base}/train/{model}", eval_payload)
    eval_job_id = r["job_id"]
    ok(f"eval job started: {eval_job_id}")
    eval_job = wait_for_job(base, eval_job_id)

    # Verify the log shows a per-job temp dir path, not the real grobid-home.
    # Wapiti logs the symlink path (model.wapiti inside the job dir) rather than
    # the resolved target, so we check for the job-dir prefix instead of the
    # exact timestamped filename.
    log_lines = eval_job.get("log", "").splitlines()
    model_path_lines = [l for l in log_lines if "Model path:" in l or "Loading model:" in l]
    if not model_path_lines:
        fail(
            "No 'Model path:' or 'Loading model:' line found in eval log — "
            "cannot confirm which model was used.\n"
            "Last 10 log lines:\n"
            + "\n".join("    " + l for l in log_lines[-10:])
        )
    # All model-path lines must reference a per-job dir (contains "grobid-job-"),
    # not the permanent grobid-home, which would mean the override was bypassed.
    wrong = [l for l in model_path_lines if "grobid-job-" not in l]
    if wrong:
        fail(
            "Eval log shows real grobid-home instead of per-job override dir — "
            f"the model_file='{trained_file}' override was not applied:\n"
            + "\n".join("    " + l for l in wrong)
        )
    ok(f"eval log confirms per-job model override (model.wapiti → {trained_file})")

    # Print evaluation metrics from the log.
    in_results = False
    for line in log_lines:
        if line.startswith("===== ") or in_results:
            in_results = True
            print(f"  {line}")

    # ── 5. Delete the timestamped file ───────────────────────────────────────
    step("5/6  Delete the timestamped model file")
    params = urllib.parse.urlencode({"name": trained_file, "flavor": flavor})
    deleted = _delete(f"{base}/models/{model}?{params}")
    if deleted.get("deleted") != trained_file:
        fail(f"unexpected delete response: {deleted}")
    ok(f"deleted: {trained_file}")
    _cleanup["trained_file"] = None  # already deleted, no cleanup needed

    # ── 6. Verify file is gone ────────────────────────────────────────────────
    step("6/6  Verify file is no longer listed")
    params = f"?flavor={urllib.parse.quote(flavor)}" if flavor else ""
    listing = _get(f"{base}/models/{model}{params}")
    names = [f["name"] for f in listing["files"]]
    if trained_file in names:
        fail(f"'{trained_file}' still appears in listing after delete: {names}")
    if "model.wapiti" not in names:
        fail("model.wapiti disappeared — something went wrong")
    ok(f"{trained_file} is gone  (model.wapiti still present)")

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  ALL STEPS PASSED")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
