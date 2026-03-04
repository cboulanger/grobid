#!/usr/bin/env python3
"""CLI client for the GROBID trainer FastAPI service (port 8072).

No third-party dependencies required — stdlib only.

Usage examples:
  python training-api-client.py health
  python training-api-client.py models
  python training-api-client.py flavors [model]
  python training-api-client.py jobs
  python training-api-client.py status <job_id>
  python training-api-client.py stream <job_id>
  python training-api-client.py train date --mode 2
  python training-api-client.py train header --mode 0 --incremental --flavor article/light
  python training-api-client.py eval nlm --p2t /path/to/dataset
  python training-api-client.py eval tei --p2t /path/to/dataset --run --file-ratio 0.5
  python training-api-client.py upload segmentation ./my-data --flavor article/dh-law-footnotes
  python training-api-client.py uploads [--model segmentation] [--flavor article/light]
  python training-api-client.py revert <batch_id>
"""

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str) -> dict:
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        _die(f"HTTP {e.code} from {url}: {body}")
    except urllib.error.URLError as e:
        _die(f"Cannot reach service at {url}: {e.reason}")


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        _die(f"HTTP {e.code} from {url}: {body}")
    except urllib.error.URLError as e:
        _die(f"Cannot reach service at {url}: {e.reason}")


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

def sse_stream(url: str) -> int:
    """Stream a Server-Sent Events endpoint, printing data lines.

    Returns the exit code extracted from the terminal 'done' event,
    or 1 if the stream ends unexpectedly.
    """
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        _die(f"HTTP {e.code} from {url}: {body}")
    except urllib.error.URLError as e:
        _die(f"Cannot reach service at {url}: {e.reason}")

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
                        # Extract exit code from "exit_code=N"
                        if text.startswith("exit_code="):
                            try:
                                exit_code = int(text.split("=", 1)[1])
                            except ValueError:
                                exit_code = 1
                        return exit_code
                    else:
                        print(text, flush=True)
                elif line == "":
                    last_event = ""  # reset event type on blank line
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130

    return exit_code


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_health(base_url: str, _args) -> int:
    data = _get(f"{base_url}/health")
    print(json.dumps(data, indent=2))
    return 0


def cmd_models(base_url: str, _args) -> int:
    data = _get(f"{base_url}/health")
    for m in data.get("models", []):
        print(m)
    return 0


def cmd_flavors(base_url: str, args) -> int:
    data = _get(f"{base_url}/flavors")
    if not data:
        print("No flavors found.")
        return 0
    if args.model:
        flavors = data.get(args.model)
        if flavors is None:
            _die(f"No flavors for model '{args.model}'")
        for f in flavors:
            print(f)
    else:
        for model, flavors in sorted(data.items()):
            for f in flavors:
                print(f"{model}\t{f}")
    return 0


def cmd_jobs(base_url: str, _args) -> int:
    jobs = _get(f"{base_url}/jobs")
    if not jobs:
        print("No jobs found.")
        return 0
    # Simple table
    col_id    = max(len("JOB ID"),  max(len(j.get("job_id", "")) for j in jobs))
    col_st    = max(len("STATUS"),  max(len(j.get("status", "")) for j in jobs))
    col_time  = max(len("STARTED"),  max(len((j.get("start_time") or "")[:19]) for j in jobs))

    header = f"{'JOB ID':<{col_id}}  {'STATUS':<{col_st}}  {'STARTED':<{col_time}}  CMD"
    print(header)
    print("-" * (len(header) + 20))
    for j in jobs:
        job_id    = j.get("job_id", "")
        status    = j.get("status", "")
        started   = (j.get("start_time") or "")[:19]
        cmd       = j.get("cmd", "")[:80]
        print(f"{job_id:<{col_id}}  {status:<{col_st}}  {started:<{col_time}}  {cmd}")
    return 0


def cmd_status(base_url: str, args) -> int:
    job = _get(f"{base_url}/jobs/{args.job_id}")
    status   = job.get("status", "?")
    duration = job.get("duration_s")
    exit_code = job.get("exit_code")

    parts = [f"[{status}]"]
    if duration is not None:
        parts.append(f"duration={duration:.1f}s")
    if exit_code is not None:
        parts.append(f"exit_code={exit_code}")
    print("  ".join(parts))

    if args.verbose:
        print(json.dumps(job, indent=2))
    elif status in ("done", "failed") and job.get("log"):
        # Print last 20 log lines for a quick summary
        lines = job["log"].splitlines()
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} lines omitted, use --verbose for full log)")
        for ln in lines[-20:]:
            print(f"  {ln}")
    return 0 if (exit_code == 0 or status == "running") else 1


def cmd_stream(base_url: str, args) -> int:
    return sse_stream(f"{base_url}/jobs/{args.job_id}/stream")


def cmd_train(base_url: str, args) -> int:
    payload = {
        "mode":        args.mode,
        "seg_ratio":   args.seg_ratio,
        "n_folds":     args.n_folds,
        "incremental": args.incremental,
        "flavor":      args.flavor,
    }
    result = _post(f"{base_url}/train/{args.model}", payload)
    job_id = result.get("job_id", "?")
    status = result.get("status", "?")
    print(f"job_id={job_id}  status={status}  model={args.model}  mode={args.mode}")

    if not args.no_stream:
        return sse_stream(f"{base_url}/jobs/{job_id}/stream")
    return 0


def cmd_eval(base_url: str, args) -> int:
    payload = {
        "p2t":        args.p2t,
        "run":        args.run,
        "file_ratio": args.file_ratio,
        "flavor":     args.flavor,
    }
    result = _post(f"{base_url}/evaluate/{args.eval_type}", payload)
    job_id = result.get("job_id", "?")
    status = result.get("status", "?")
    print(f"job_id={job_id}  status={status}  type={args.eval_type}")

    if not args.no_stream:
        return sse_stream(f"{base_url}/jobs/{job_id}/stream")
    return 0


# ---------------------------------------------------------------------------
# Multipart/form-data helper (stdlib only)
# ---------------------------------------------------------------------------

def _multipart_post(url: str, fields: dict, file_field: str, filename: str, file_bytes: bytes) -> dict:
    """POST multipart/form-data with text fields + one file field."""
    boundary = uuid.uuid4().hex
    ctype    = f"multipart/form-data; boundary={boundary}"

    body = io.BytesIO()
    enc  = "utf-8"

    for name, value in fields.items():
        body.write(f"--{boundary}\r\n".encode(enc))
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(enc))
        body.write(f"{value}\r\n".encode(enc))

    body.write(f"--{boundary}\r\n".encode(enc))
    body.write(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode(enc)
    )
    body.write(b"Content-Type: application/zip\r\n\r\n")
    body.write(file_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode(enc))

    data = body.getvalue()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        _die(f"HTTP {e.code} from {url}: {body_text}")
    except urllib.error.URLError as e:
        _die(f"Cannot reach service at {url}: {e.reason}")


# ---------------------------------------------------------------------------
# Upload / uploads / revert commands
# ---------------------------------------------------------------------------

def cmd_upload(base_url: str, args) -> int:
    src = Path(args.directory).expanduser().resolve()
    if not src.is_dir():
        _die(f"Not a directory: {src}")

    # Build ZIP in memory from directory contents (flat — filenames only)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in sorted(src.rglob("*")):
            if entry.is_file():
                zf.write(entry, entry.name)
    zip_bytes = buf.getvalue()
    print(f"Zipped {src} → {len(zip_bytes):,} bytes", file=sys.stderr)

    fields = {"flavor": args.flavor, "batch_name": args.batch_name}
    result = _multipart_post(
        f"{base_url}/upload/{args.model}",
        fields,
        file_field="file",
        filename=f"{args.model}.zip",
        file_bytes=zip_bytes,
    )

    batch_id = result.get("batch_id", "?")
    added    = result.get("files_added_count", 0)
    skipped  = result.get("files_skipped_count", 0)
    print(f"batch_id={batch_id}  files_added={added}  files_skipped={skipped}")
    if args.verbose:
        if result.get("files_added"):
            print("  Added:")
            for f in result["files_added"]:
                print(f"    {f}")
        if result.get("files_skipped"):
            print("  Skipped (already exist):")
            for f in result["files_skipped"]:
                print(f"    {f}")
    return 0


def cmd_uploads(base_url: str, args) -> int:
    params = {}
    if args.model:
        params["model"] = args.model
    if args.flavor:
        params["flavor"] = args.flavor
    qs  = ("?" + urllib.parse.urlencode(params)) if params else ""
    batches = _get(f"{base_url}/uploads{qs}")
    if not batches:
        print("No upload batches found.")
        return 0
    col_id   = max(len("BATCH ID"),   max(len(b.get("batch_id", ""))   for b in batches))
    col_mod  = max(len("MODEL"),      max(len(b.get("model", ""))       for b in batches))
    col_flv  = max(len("FLAVOR"),     max(len(b.get("flavor", ""))      for b in batches))
    col_ts   = max(len("TIMESTAMP"),  max(len(b.get("timestamp", "")[:19]) for b in batches))
    header = (f"{'BATCH ID':<{col_id}}  {'MODEL':<{col_mod}}  {'FLAVOR':<{col_flv}}"
              f"  {'TIMESTAMP':<{col_ts}}  ADDED  SKIPPED  NAME")
    print(header)
    print("-" * (len(header) + 10))
    for b in batches:
        print(
            f"{b.get('batch_id',''):<{col_id}}  "
            f"{b.get('model',''):<{col_mod}}  "
            f"{b.get('flavor',''):<{col_flv}}  "
            f"{b.get('timestamp','')[:19]:<{col_ts}}  "
            f"{b.get('files_added_count',0):>5}  "
            f"{b.get('files_skipped_count',0):>7}  "
            f"{b.get('batch_name','')}"
        )
    return 0


def cmd_revert(base_url: str, args) -> int:
    result   = _post(f"{base_url}/revert/{args.batch_id}", {})
    removed  = result.get("files_removed", [])
    missing  = result.get("files_missing", [])
    print(f"batch_id={args.batch_id}  files_removed={len(removed)}  files_missing={len(missing)}")
    if args.verbose:
        for f in removed:
            print(f"  removed: {f}")
        for f in missing:
            print(f"  missing: {f}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

MODES_HELP = (
    "0=train only, 1=evaluate only, "
    "2=auto-split train+eval (uses --seg-ratio), "
    "3=n-fold cross-eval (uses --n-folds)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training-api-client.py",
        description="CLI client for the GROBID trainer service (default port 8072).",
    )
    parser.add_argument("--host", default="localhost", help="Service host (default: localhost)")
    parser.add_argument("--port", type=int, default=8072, help="Service port (default: 8072)")

    sub = parser.add_subparsers(dest="command", required=True)

    # health
    sub.add_parser("health", help="Check service health and environment")

    # models
    sub.add_parser("models", help="List available models reported by the service")

    # flavors
    p_flavors = sub.add_parser("flavors", help="List dataset flavors per model")
    p_flavors.add_argument("model", nargs="?", default="",
                           help="Filter to a specific model (e.g. header, segmentation)")

    # jobs
    sub.add_parser("jobs", help="List all jobs")

    # status
    p_status = sub.add_parser("status", help="Get status of a specific job")
    p_status.add_argument("job_id", help="Job ID")
    p_status.add_argument("-v", "--verbose", action="store_true",
                          help="Print full JSON response and complete log")

    # stream
    p_stream = sub.add_parser("stream", help="Stream live log of a running job")
    p_stream.add_argument("job_id", help="Job ID")

    # train
    p_train = sub.add_parser("train", help="Start model training")
    p_train.add_argument("model", help="Model to train (e.g. date, header, citation-BidLSTM_CRF_FEATURES)")
    p_train.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3],
                         metavar="MODE", help=MODES_HELP)
    p_train.add_argument("--seg-ratio", type=float, default=0.8,
                         help="Train/eval split ratio for mode 2 (default: 0.8)")
    p_train.add_argument("--n-folds", type=int, default=10,
                         help="Number of folds for mode 3 (default: 10)")
    p_train.add_argument("--incremental", action="store_true",
                         help="Start from existing model instead of training from scratch")
    p_train.add_argument("--flavor", default="",
                         help="Model variant, e.g. 'article/light'")
    p_train.add_argument("--no-stream", action="store_true",
                         help="Submit job and return immediately without streaming the log")

    # eval
    p_eval = sub.add_parser("eval", help="Run end-to-end evaluation")
    p_eval.add_argument("eval_type", choices=["nlm", "tei"],
                        help="Evaluation format: nlm or tei")
    p_eval.add_argument("--p2t", required=True,
                        help="Absolute path to gold-standard dataset directory")
    p_eval.add_argument("--run", action="store_true",
                        help="Run GROBID on PDFs first (requires GROBID service on port 8070)")
    p_eval.add_argument("--file-ratio", type=float, default=1.0,
                        help="Fraction of dataset files to use, 0.0–1.0 (default: 1.0)")
    p_eval.add_argument("--flavor", default="", help="Model variant, e.g. 'article/light'")
    p_eval.add_argument("--no-stream", action="store_true",
                        help="Submit job and return immediately without streaming the log")

    # upload
    p_upload = sub.add_parser("upload", help="Upload training data ZIP (non-destructive)")
    p_upload.add_argument("model", help="Target model (e.g. segmentation, header)")
    p_upload.add_argument("directory", help="Local directory whose files will be zipped and uploaded")
    p_upload.add_argument("--flavor", default="", help="Model flavor, e.g. 'article/dh-law-footnotes'")
    p_upload.add_argument("--batch-name", default="", help="Optional human-readable label for this batch")
    p_upload.add_argument("-v", "--verbose", action="store_true", help="List individual added/skipped files")

    # uploads
    p_uploads = sub.add_parser("uploads", help="List upload batches")
    p_uploads.add_argument("--model",  default="", help="Filter by model name")
    p_uploads.add_argument("--flavor", default="", help="Filter by flavor")

    # revert
    p_revert = sub.add_parser("revert", help="Revert an upload batch (removes added files)")
    p_revert.add_argument("batch_id", help="Batch ID to revert")
    p_revert.add_argument("-v", "--verbose", action="store_true", help="List individual removed/missing files")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    dispatch = {
        "health":  cmd_health,
        "models":  cmd_models,
        "flavors": cmd_flavors,
        "jobs":    cmd_jobs,
        "status":  cmd_status,
        "stream":  cmd_stream,
        "train":   cmd_train,
        "eval":    cmd_eval,
        "upload":  cmd_upload,
        "uploads": cmd_uploads,
        "revert":  cmd_revert,
    }

    handler = dispatch[args.command]
    sys.exit(handler(base_url, args))


if __name__ == "__main__":
    main()
