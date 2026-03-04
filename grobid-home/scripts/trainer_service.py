#!/usr/bin/env python3
"""
GROBID Trainer HTTP Service
============================
Wraps grobid-trainer for model training and end-to-end evaluation via REST API.
Designed for Apple Silicon (M4) running inside `nix develop` from flake.nix.

Usage:
    python grobid-home/scripts/trainer_service.py [--port 8072] [--host 0.0.0.0]

Prerequisites:
    1. Run `nix develop` (or activate your conda/venv with tensorflow-metal + jep)
    2. Build the trainer JAR: ./gradlew :grobid-trainer:shadowJar --no-daemon

Endpoints:
    POST /train/{model}         Start training a model
    POST /evaluate/{nlm|tei}    Run end-to-end evaluation
    GET  /jobs/{job_id}         Job status + full log
    GET  /jobs/{job_id}/stream  Live log via SSE
    GET  /jobs                  List all jobs
    GET  /health                Environment info

Notes on evaluation:
    The /evaluate endpoints invoke EndToEndEvaluation which can optionally
    run GROBID to process a PDF gold set (run=true). In that case a running
    GROBID service is required (default http://localhost:8070). When run=false
    the evaluation uses already-processed output in the p2t directory.
"""

import argparse
import glob
import os
import platform
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent   # grobid-home/scripts/
GROBID_HOME = SCRIPT_DIR.parent.resolve()       # grobid-home/
GROBID_ROOT = GROBID_HOME.parent.resolve()      # project root


def find_trainer_jar() -> Optional[Path]:
    """Locate the pre-built grobid-trainer shadow (onejar) JAR."""
    pattern = str(GROBID_ROOT / "grobid-trainer" / "build" / "libs" / "grobid-trainer-*-onejar.jar")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _jep_path_from_env(env_prefix: str) -> Optional[str]:
    """Return the JEP site-packages directory inside a venv/conda prefix."""
    lib_dir = Path(env_prefix) / "lib"
    if not lib_dir.exists():
        return None
    for python_dir in lib_dir.glob("python*"):
        jep_dir = python_dir / "site-packages" / "jep"
        if jep_dir.exists():
            return str(jep_dir)
    return None


def build_java_lib_path() -> str:
    """
    Construct java.library.path for the current platform.

    On macOS ARM64 (Apple Silicon):
      - Wapiti: grobid-home/lib/mac_arm-64/libwapiti.dylib
      - JEP:    <venv>/lib/python3.x/site-packages/jep/libjep.dylib

    The VIRTUAL_ENV / CONDA_PREFIX detection mirrors build.gradle's
    getJavaLibraryPath() so both Gradle tasks and this service resolve
    the same paths.
    """
    parts: List[str] = []
    machine = platform.machine().lower()   # 'arm64' on M-series Macs
    system  = platform.system().lower()    # 'darwin' on macOS

    if system == "darwin":
        lib_subdir = "mac_arm-64" if machine in ("arm64", "aarch64") else "mac-64"
        parts.append(str(GROBID_HOME / "lib" / lib_subdir))
    else:
        # Linux (e.g. CI, Docker ARM64)
        if machine in ("arm64", "aarch64"):
            parts.append(str(GROBID_HOME / "lib" / "lin_arm-64" / "jep"))
            parts.append(str(GROBID_HOME / "lib" / "lin_arm-64"))
        else:
            parts.append(str(GROBID_HOME / "lib" / "lin-64" / "jep"))
            parts.append(str(GROBID_HOME / "lib" / "lin-64"))

    # DeLFT: locate JEP dylib from active virtual/conda environment
    for env_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        prefix = os.environ.get(env_var)
        if prefix:
            jep = _jep_path_from_env(prefix)
            if jep:
                parts.append(jep)
            break

    return ":".join(p for p in parts if p)


def _base_java_cmd(jar: Path, *, use_cp: bool = False) -> List[str]:
    """Base JVM invocation with memory, module opens, and library path."""
    lib_path = build_java_lib_path()
    cmd = [
        "java",
        "-Xmx4g",
        f"-Djava.library.path={lib_path}",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-opens", "java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens", "java.base/java.io=ALL-UNNAMED",
    ]
    if use_cp:
        cmd += ["-cp", str(jar)]
    else:
        cmd += ["-jar", str(jar)]
    return cmd


# ── Job management ─────────────────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _stream_proc(job_id: str, proc: subprocess.Popen) -> None:
    """Read subprocess output and append to job log; update final status."""
    job = _jobs[job_id]
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            with _jobs_lock:
                job["log"].append(line.rstrip())
        proc.wait()
        with _jobs_lock:
            job["exit_code"] = proc.returncode
            job["status"]    = "done" if proc.returncode == 0 else "failed"
    except Exception as exc:
        with _jobs_lock:
            job["log"].append(f"[service error] {exc}")
            job["status"]    = "failed"
            job["exit_code"] = -1
    finally:
        with _jobs_lock:
            job["end_time"] = datetime.utcnow().isoformat()


def _start_job(cmd: List[str]) -> str:
    job_id = str(uuid.uuid4())[:8]
    job: Dict[str, Any] = {
        "job_id":     job_id,
        "status":     "running",
        "exit_code":  None,
        "log":        [],
        "start_time": datetime.utcnow().isoformat(),
        "end_time":   None,
        "pid":        None,
        "cmd":        " ".join(cmd),
    }
    with _jobs_lock:
        _jobs[job_id] = job

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(GROBID_ROOT),
        bufsize=1,
    )
    job["pid"] = proc.pid
    threading.Thread(target=_stream_proc, args=(job_id, proc), daemon=True).start()
    return job_id


# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GROBID Trainer Service",
    description=(
        "REST API for GROBID model training and evaluation on Apple Silicon. "
        "Run inside `nix develop` (see flake.nix) for Metal GPU support."
    ),
    version="1.0.0",
)

VALID_MODELS = {
    "date", "header", "name-header", "name-citation", "affiliation-address",
    "citation", "segmentation", "fulltext", "reference-segmentation", "figure",
    "table", "patent-citation", "funding-acknowledgement", "shorttext", "ebook-model",
}


class TrainRequest(BaseModel):
    mode: int = Field(
        default=0,
        description=(
            "0 = train only  |  1 = evaluate only  |  "
            "2 = auto-split then train+evaluate  |  3 = N-fold cross-evaluation"
        ),
    )
    seg_ratio: float = Field(default=0.8,  description="Train/eval split ratio (mode 2 only)")
    n_folds:   int   = Field(default=10,   description="Number of folds (mode 3 only)")
    incremental: bool = Field(default=False, description="Incremental training from existing model")
    flavor: str = Field(
        default="",
        description="Model flavour variant, e.g. 'article/light', 'article/light-ref', 'sdo/ietf'",
    )


class EvalRequest(BaseModel):
    p2t: str = Field(description="Absolute path to the gold-standard dataset directory")
    run: bool = Field(
        default=False,
        description=(
            "If true, run GROBID on the PDFs in p2t (requires GROBID service on port 8070). "
            "If false, evaluate already-processed output files."
        ),
    )
    file_ratio: float = Field(default=1.0, description="Fraction of the dataset to use (0.0–1.0)")
    flavor: str = Field(default="", description="Model flavour variant")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/train/{model_name}",
    summary="Start model training",
    response_description="Job ID and initial status",
)
def train(model_name: str, req: TrainRequest):
    """
    Trigger training (or evaluation) of a GROBID model.

    The `mode` field controls the operation:
    - **0** – Train using all files in `grobid-trainer/resources/dataset/{model}/corpus/`
    - **1** – Evaluate using files in `grobid-trainer/resources/dataset/{model}/evaluation/`
    - **2** – Randomly split corpus by `seg_ratio`, train, then evaluate
    - **3** – N-fold cross-evaluation using `n_folds` folds

    Training data must be present in the corresponding dataset directory.
    Trained models are written to `grobid-home/models/{model}/`.
    """
    if model_name not in VALID_MODELS:
        raise HTTPException(
            400,
            f"Unknown model '{model_name}'. Valid models: {sorted(VALID_MODELS)}",
        )
    if req.mode not in (0, 1, 2, 3):
        raise HTTPException(400, "mode must be 0, 1, 2, or 3")

    jar = find_trainer_jar()
    if jar is None:
        raise HTTPException(
            503,
            "Trainer JAR not found. Build it first:\n"
            "  ./gradlew :grobid-trainer:shadowJar --no-daemon",
        )

    cmd = _base_java_cmd(jar) + [str(req.mode), model_name, "-gH", str(GROBID_HOME)]
    if req.mode == 2:
        cmd += ["-s", str(req.seg_ratio)]
    if req.mode == 3:
        cmd += ["-n", str(req.n_folds)]
    if req.incremental:
        cmd += ["-i"]
    if req.flavor:
        cmd.append(req.flavor)

    job_id = _start_job(cmd)
    return {"job_id": job_id, "status": "running", "model": model_name, "mode": req.mode}


@app.post(
    "/evaluate/{eval_type}",
    summary="Run end-to-end evaluation",
    response_description="Job ID and initial status",
)
def evaluate(eval_type: str, req: EvalRequest):
    """
    Run GROBID end-to-end evaluation against a gold-standard dataset.

    - **nlm** – evaluate against NLM/JATS-formatted gold XML
    - **tei** – evaluate against TEI-formatted gold XML

    When `run=true`, GROBID is invoked on the PDF files in `p2t` via the
    running GROBID service (default port 8070). Start the main GROBID service
    first and make sure `p2t` contains the PDF gold set.

    When `run=false`, evaluation uses existing GROBID output already present
    in `p2t` (useful to re-score without reprocessing).

    The evaluation report is written to `grobid-home/tmp/report.md`.
    """
    if eval_type not in ("nlm", "tei"):
        raise HTTPException(400, "eval_type must be 'nlm' or 'tei'")

    jar = find_trainer_jar()
    if jar is None:
        raise HTTPException(
            503,
            "Trainer JAR not found. Build it first:\n"
            "  ./gradlew :grobid-trainer:shadowJar --no-daemon",
        )

    cmd = _base_java_cmd(jar, use_cp=True) + [
        "org.grobid.trainer.evaluation.EndToEndEvaluation",
        eval_type,
        req.p2t,
        "1" if req.run else "0",
        str(req.file_ratio),
        req.flavor,
    ]

    job_id = _start_job(cmd)
    return {"job_id": job_id, "status": "running", "type": eval_type, "p2t": req.p2t}


@app.get("/jobs/{job_id}", summary="Get job status and log")
def get_job(job_id: str):
    """Return the current status, exit code, and full captured log for a job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found")

    duration: Optional[float] = None
    if job.get("end_time"):
        try:
            start = datetime.fromisoformat(job["start_time"])
            end   = datetime.fromisoformat(job["end_time"])
            duration = (end - start).total_seconds()
        except Exception:
            pass

    return {
        **{k: v for k, v in job.items() if k != "log"},
        "log":        "\n".join(job["log"]),
        "duration_s": duration,
    }


@app.get("/jobs/{job_id}/stream", summary="Stream job log via Server-Sent Events")
def stream_job(job_id: str):
    """
    Stream live output from a running (or completed) job as Server-Sent Events.

    Connect with:
        curl -N http://localhost:8072/jobs/{job_id}/stream

    The stream ends with an `event: done` message containing the exit code.
    """
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(404, f"Job '{job_id}' not found")

    def generate() -> Iterator[str]:
        seen = 0
        while True:
            with _jobs_lock:
                job   = _jobs[job_id]
                lines = job["log"]
                new   = lines[seen:]
                done  = job["status"] != "running"

            for line in new:
                yield f"data: {line}\n\n"
            seen += len(new)

            if done and seen >= len(_jobs[job_id]["log"]):
                exit_code = _jobs[job_id]["exit_code"]
                yield f"event: done\ndata: exit_code={exit_code}\n\n"
                return

            time.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/jobs", summary="List all jobs")
def list_jobs():
    """Return a summary list of all submitted jobs."""
    with _jobs_lock:
        return [
            {
                "job_id":     j["job_id"],
                "status":     j["status"],
                "start_time": j["start_time"],
                "cmd":        j["cmd"][:120],
            }
            for j in _jobs.values()
        ]


@app.get("/health", summary="Service health and environment info")
def health():
    """
    Check service health and report environment paths.

    Useful to verify that the JAR has been built, the native library path
    is correct, and the expected GROBID models are present.
    """
    jar = find_trainer_jar()
    models_dir = GROBID_HOME / "models"
    available_models = (
        sorted(p.name for p in models_dir.iterdir() if p.is_dir())
        if models_dir.exists()
        else []
    )

    return {
        "status":       "ok",
        "platform":     f"{platform.system()}/{platform.machine()}",
        "grobid_root":  str(GROBID_ROOT),
        "grobid_home":  str(GROBID_HOME),
        "jar":          str(jar) if jar else None,
        "jar_built":    jar is not None,
        "java_lib_path": build_java_lib_path(),
        "virtual_env":  os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "models":       available_models,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GROBID Trainer HTTP Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8072, help="Bind port (default: 8072)")
    args = parser.parse_args()

    jar = find_trainer_jar()

    print(f"GROBID Trainer Service")
    print(f"  GROBID root  : {GROBID_ROOT}")
    print(f"  Platform     : {platform.system()}/{platform.machine()}")
    print(f"  java.lib.path: {build_java_lib_path()}")
    print(f"  Trainer JAR  : {jar or 'NOT FOUND — run ./gradlew :grobid-trainer:shadowJar --no-daemon'}")
    print(f"  API docs     : http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(app, host=args.host, port=args.port)
