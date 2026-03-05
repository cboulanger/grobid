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
    POST /jobs/{job_id}/stop    Stop a running job
    GET  /jobs                  List all jobs
    GET  /flavors               Flavors per model (from dataset dirs)
    POST /upload/{model}        Upload ZIP of training data (non-destructive)
    GET  /uploads               List upload batches
    GET  /uploads/{batch_id}    Get upload batch details
    POST /revert/{batch_id}     Revert an upload batch
    GET  /models/{model}        List model files (wapiti variants) for a model
    DELETE /models/{model}      Delete a specific model file by name
    GET  /health                Environment info

Notes on evaluation:
    The /evaluate endpoints invoke EndToEndEvaluation which can optionally
    run GROBID to process a PDF gold set (run=true). In that case a running
    GROBID service is required (default http://localhost:8070). When run=false
    the evaluation uses already-processed output in the p2t directory.
"""

import argparse
import glob
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


def _base_java_cmd(jar: Path, *, use_cp: bool = False, grobid_home: Optional[Path] = None) -> List[str]:
    """Base JVM invocation with memory, module opens, and library path.

    Pass grobid_home to override the grobid home via -Dorg.grobid.home.
    GrobidHomeFinder checks this system property first, making it the only
    reliable way to redirect GROBID to a per-job home directory.
    (The -gH CLI flag is parsed by TrainerRunner but never applied to
    GrobidProperties — it is effectively ignored.)
    """
    lib_path = build_java_lib_path()
    cmd = [
        "java",
        "-Xmx4g",
        f"-Djava.library.path={lib_path}",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-opens", "java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens", "java.base/java.io=ALL-UNNAMED",
    ]
    if grobid_home is not None:
        cmd.append(f"-Dorg.grobid.home={grobid_home}")
    if use_cp:
        cmd += ["-cp", str(jar)]
    else:
        cmd += ["-jar", str(jar)]
    return cmd


# ── Per-job grobid-home ────────────────────────────────────────────────────────

def _make_per_job_grobid_home(
    job_id: str,
    model_name: str,
    flavor: str,
    epsilon: Optional[float],
    nb_max_iterations: Optional[int],
    model_file: Optional[str] = None,
) -> Optional[Path]:
    """
    Create a temporary grobid-home for a single training job.
    Returns None if no overrides are needed (all parameters are None).

    Always writes a patched grobid.yaml with grobidHome set to the temp dir's
    absolute path.  Without this, GROBID resolves the relative "grobid-home"
    value from the process working directory and ignores -gH, so model paths
    always point at the real grobid-home regardless of the temp dir structure.

    The temp dir is created inside GROBID_ROOT (not /tmp/) because
    AbstractTrainer.getFilePath2Resources() resolves the training dataset as
    {grobidHome}/../grobid-trainer/resources — it must be a sibling of grobid-trainer/.

    Structure:
        {tmpdir}/config/grobid.yaml   ← patched copy (grobidHome + optional Wapiti params)
        {tmpdir}/models/              ← symlink to real models dir, or overridden subtree
        {tmpdir}/<everything else>    ← symlinks to real grobid-home

    The caller is responsible for deleting the directory when the job finishes.
    """
    if epsilon is None and nb_max_iterations is None and model_file is None:
        return None

    # ── Build temporary grobid-home ────────────────────────────────────────────
    # Create the temp dir first so we know its absolute path, which must be
    # written into grobid.yaml (grobidHome).  Without this, GROBID resolves
    # the relative "grobid-home" value from the working directory and ignores
    # the -gH flag entirely, meaning all model overrides have no effect.
    #
    # Create the temp dir inside GROBID_ROOT (not /tmp/) so that
    # AbstractTrainer.getFilePath2Resources() can find the training dataset via
    # {grobidHome}/../grobid-trainer/resources (it walks one level up from grobidHome).
    tmp_home = Path(tempfile.mkdtemp(prefix=f"grobid-job-{job_id}-", dir=GROBID_ROOT))

    # ── Yaml config: always write a patched copy ───────────────────────────────
    config_path = GROBID_HOME / "config" / "grobid.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Point grobidHome at the temp dir so model paths resolve through it.
    if "grobid" not in config:
        config["grobid"] = {}
    config["grobid"]["grobidHome"] = str(tmp_home)

    # Optionally patch Wapiti training parameters.
    if epsilon is not None or nb_max_iterations is not None:
        yaml_model_key = model_name if not flavor else f"{model_name}-{flavor.replace('/', '-')}"
        patched = False
        for m in config.get("grobid", {}).get("models", []):
            if m.get("name") == yaml_model_key:
                if m.get("wapiti") is None:
                    m["wapiti"] = {}
                if epsilon is not None:
                    m["wapiti"]["epsilon"] = epsilon
                if nb_max_iterations is not None:
                    m["wapiti"]["nbMaxIterations"] = nb_max_iterations
                patched = True
                break
        if not patched:
            print(
                f"WARNING: model '{yaml_model_key}' not found in grobid.yaml — "
                "epsilon/nb_max_iterations overrides ignored.",
                flush=True,
            )

    config_dir = tmp_home / "config"
    config_dir.mkdir()
    with (config_dir / "grobid.yaml").open("w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Symlink every other entry from the real grobid-home (models handled below)
    for entry in GROBID_HOME.iterdir():
        if entry.name in ("config", "models"):
            continue
        (tmp_home / entry.name).symlink_to(entry.resolve())

    # Models: override specific model file, or symlink whole models dir
    if model_file:
        _override_model_dir(tmp_home, model_name, flavor, model_file)
    else:
        (tmp_home / "models").symlink_to((GROBID_HOME / "models").resolve())

    return tmp_home


# ── Model directory helpers ───────────────────────────────────────────────────

def _model_dir(model_name: str, flavor: str) -> Path:
    """Return the model directory for a model + optional flavor."""
    base = GROBID_HOME / "models" / model_name
    return base / flavor if flavor else base


def _post_training_rename(model_name: str, flavor: str, keep_existing: bool) -> str:
    """
    Post-process model files after a successful training run.

    The JAR always leaves:
        model.wapiti       — newly trained model
        model.wapiti.old   — previous model (may not exist on first train)

    keep_existing=False (default):
        Rename model.wapiti.old → model.wapiti.{timestamp}
        Active model stays as model.wapiti (new model replaces old)

    keep_existing=True:
        Rename model.wapiti     → model.wapiti.{timestamp}  (new, for inspection)
        Rename model.wapiti.old → model.wapiti              (restore previous active)

    Returns the filename of the newly trained model.
    """
    mdir    = _model_dir(model_name, flavor)
    ts      = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    active  = mdir / "model.wapiti"
    old     = mdir / "model.wapiti.old"
    ts_name = f"model.wapiti.{ts}"
    ts_path = mdir / ts_name

    if not mdir.exists():
        print(
            f"[service warning] post-training rename skipped: model directory not found: {mdir}",
            flush=True,
        )
        return "model.wapiti"

    if keep_existing:
        if active.exists():
            active.rename(ts_path)      # new model → timestamped
        if old.exists():
            old.rename(active)          # previous model restored as active
        return ts_path.resolve().name
    else:
        if old.exists():
            old.rename(ts_path)         # previous model → timestamped (replaces .old)
        return (mdir / "model.wapiti").resolve().name


def _override_model_dir(tmp_home: Path, model_name: str, flavor: str, model_file: str) -> None:
    """
    Build a models/ subtree inside tmp_home where the target model's
    model.wapiti symlinks to the specified model_file instead of the real one.

    All other model dirs and files are symlinked from the real grobid-home.
    """
    real_models = GROBID_HOME / "models"
    tmp_models  = tmp_home / "models"
    tmp_models.mkdir()

    # Symlink every top-level model dir except our target
    for entry in real_models.iterdir():
        if entry.name != model_name:
            (tmp_models / entry.name).symlink_to(entry.resolve())

    # Walk down model_name[/flavor] creating real dirs and symlinking siblings
    parts    = [model_name] + (flavor.split("/") if flavor else [])
    real_cur = real_models
    tmp_cur  = tmp_models
    for i, part in enumerate(parts):
        real_cur = real_cur / part
        tmp_cur  = tmp_cur / part
        tmp_cur.mkdir(exist_ok=True)
        is_leaf = (i == len(parts) - 1)
        for entry in real_cur.iterdir():
            if is_leaf and entry.name == "model.wapiti":
                continue    # replaced below
            (tmp_cur / entry.name).symlink_to(entry.resolve())

    # Override model.wapiti with the requested file
    target = real_models.joinpath(*parts) / model_file
    if not target.exists():
        raise FileNotFoundError(f"Model file not found: {target}")
    (tmp_cur / "model.wapiti").symlink_to(target.resolve())


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
            if job["status"] != "cancelled":   # don't overwrite a user-requested stop
                job["status"] = "done" if proc.returncode == 0 else "failed"
    except Exception as exc:
        with _jobs_lock:
            job["log"].append(f"[service error] {exc}")
            job["status"]    = "failed"
            job["exit_code"] = -1
    finally:
        with _jobs_lock:
            job["end_time"] = datetime.now(timezone.utc).isoformat()
        # Post-training: rename model files (timestamped backup, keep_existing swap)
        if job.get("status") == "done" and job.get("mode") in (0, 2, 3) and job.get("model"):
            try:
                trained_file = _post_training_rename(
                    job["model"], job.get("flavor", ""), job.get("keep_existing", False)
                )
                with _jobs_lock:
                    job["trained_model_file"] = trained_file
            except Exception as exc:
                with _jobs_lock:
                    job["log"].append(f"[service warning] post-training rename failed: {exc}")
        tmp_home = job.get("_tmp_home")
        if tmp_home:
            shutil.rmtree(tmp_home, ignore_errors=True)


def _start_job(cmd: List[str], **meta: Any) -> str:
    job_id = meta.pop("job_id", None) or str(uuid.uuid4())[:8]
    job: Dict[str, Any] = {
        "job_id":     job_id,
        "status":     "running",
        "exit_code":  None,
        "log":        [],
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time":   None,
        "pid":        None,
        "cmd":        " ".join(cmd),
        **meta,
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
    job["pid"]  = proc.pid
    job["proc"] = proc          # kept for stop endpoint; not exposed in API responses
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
    epsilon: Optional[float] = Field(
        default=None,
        description=(
            "Wapiti L-BFGS convergence threshold. Overrides the value in grobid.yaml for this job only. "
            "Smaller = more iterations = longer training, higher accuracy. "
            "Suggested values: 0.001 (fast/smoke-test), 0.0001 (development), 0.00001 (production)."
        ),
    )
    nb_max_iterations: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of Wapiti training iterations. Overrides the value in grobid.yaml for this job only. "
            "Combine with a looser epsilon to cap training time during development."
        ),
    )
    keep_existing: bool = Field(
        default=False,
        description=(
            "If true, the newly trained model is saved as model.wapiti.{timestamp} and the "
            "current active model is preserved unchanged. "
            "If false (default), the new model becomes the active model.wapiti; "
            "the previous model is saved as model.wapiti.{timestamp} instead of .old."
        ),
    )
    model_file: Optional[str] = Field(
        default=None,
        description=(
            "For mode=1 (evaluate only): specific model filename to evaluate, "
            "e.g. 'model.wapiti.20260305-143022'. "
            "Omit to use the currently active model.wapiti. "
            "Ignored for modes 0, 2, and 3."
        ),
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
    model_file: Optional[str] = Field(
        default=None,
        description=(
            "Specific model filename to evaluate against, e.g. 'model.wapiti.20260305-143022'. "
            "Omit to use the currently active model.wapiti. "
            "The named file must exist in the model's directory."
        ),
    )


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

    if req.flavor:
        flavor_dataset_dir = (
            GROBID_ROOT / "grobid-trainer" / "resources" / "dataset" / model_name / req.flavor
        )
        if not flavor_dataset_dir.is_dir():
            raise HTTPException(
                400,
                f"Unknown flavor '{req.flavor}' for model '{model_name}'. "
                f"No dataset directory found at: {flavor_dataset_dir}. "
                "Use GET /flavors to list available flavors.",
            )

    jar = find_trainer_jar()
    if jar is None:
        raise HTTPException(
            503,
            "Trainer JAR not found. Build it first:\n"
            "  ./gradlew :grobid-trainer:shadowJar --no-daemon",
        )

    model_file = req.model_file if req.mode == 1 else None
    if model_file and ("/" in model_file or "\\" in model_file):
        raise HTTPException(400, "model_file must be a bare filename, not a path")

    job_id   = str(uuid.uuid4())[:8]
    tmp_home = _make_per_job_grobid_home(
        job_id, model_name, req.flavor, req.epsilon, req.nb_max_iterations, model_file
    )
    effective_home = tmp_home if tmp_home else GROBID_HOME

    cmd = _base_java_cmd(jar, grobid_home=effective_home) + [
        str(req.mode), model_name, "-gH", str(effective_home),
    ]
    if req.mode == 2:
        cmd += ["-s", str(req.seg_ratio)]
    if req.mode == 3:
        cmd += ["-n", str(req.n_folds)]
    if req.incremental:
        cmd += ["-i"]
    if req.flavor:
        cmd.append(req.flavor)

    job_id = _start_job(
        cmd,
        job_id=job_id,
        model=model_name,
        flavor=req.flavor,
        mode=req.mode,
        epsilon=req.epsilon,
        nb_max_iterations=req.nb_max_iterations,
        keep_existing=req.keep_existing,
        _tmp_home=str(tmp_home) if tmp_home else None,
    )
    return {"job_id": job_id, "status": "running", "model": model_name, "flavor": req.flavor, "mode": req.mode}


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

    eval_job_id = str(uuid.uuid4())[:8]
    if req.model_file:
        # Validate: no path separators
        if "/" in req.model_file or "\\" in req.model_file:
            raise HTTPException(400, "model_file must be a bare filename, not a path")
        # Derive model_name from eval_type isn't possible — model_file override for
        # end-to-end eval requires knowing which model to override; not supported here.
        # model_file is only meaningful for mode-1 (single-model eval via /train).
        raise HTTPException(
            400,
            "model_file is not supported for end-to-end evaluation (/evaluate). "
            "Use POST /train/{model} with mode=1 and model_file to evaluate a single model."
        )

    cmd = _base_java_cmd(jar, use_cp=True) + [
        "org.grobid.trainer.evaluation.EndToEndEvaluation",
        eval_type,
        req.p2t,
        "1" if req.run else "0",
        str(req.file_ratio),
        req.flavor,
    ]

    job_id = _start_job(cmd, job_id=eval_job_id, eval_type=eval_type, flavor=req.flavor)
    return {"job_id": job_id, "status": "running", "eval_type": eval_type, "flavor": req.flavor, "p2t": req.p2t}


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
        **{k: v for k, v in job.items() if k not in ("log", "proc", "_tmp_home")},
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
                done  = job["status"] not in ("running",)

            for line in new:
                yield f"data: {line}\n\n"
            seen += len(new)

            if done and seen >= len(_jobs[job_id]["log"]):
                exit_code = _jobs[job_id]["exit_code"]
                yield f"event: done\ndata: exit_code={exit_code}\n\n"
                return

            time.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/jobs/{job_id}/stop", summary="Stop a running job")
def stop_job(job_id: str):
    """
    Send SIGTERM to a running job's subprocess and mark it as cancelled.

    If the job is not running (already done, failed, or cancelled) a 409 is
    returned.  The job record is kept so the partial log can still be retrieved.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found")
    with _jobs_lock:
        if job["status"] != "running":
            raise HTTPException(409, f"Job '{job_id}' is not running (status: {job['status']})")
        job["status"] = "cancelled"

    proc: Optional[subprocess.Popen] = job.get("proc")
    if proc is not None:
        try:
            proc.terminate()        # SIGTERM — graceful shutdown
        except OSError:
            pass                    # process already exited between the check and here

    return {"job_id": job_id, "status": "cancelled"}


@app.get("/jobs", summary="List all jobs")
def list_jobs():
    """Return a summary list of all submitted jobs."""
    with _jobs_lock:
        return [
            {
                "job_id":             j["job_id"],
                "status":             j["status"],
                "start_time":         j["start_time"],
                "model":              j.get("model"),
                "eval_type":          j.get("eval_type"),
                "flavor":             j.get("flavor"),
                "trained_model_file": j.get("trained_model_file"),
                "cmd":                j["cmd"][:120],
            }
            for j in _jobs.values()
        ]


def _discover_flavors() -> Dict[str, List[str]]:
    """
    Walk grobid-trainer/resources/dataset/ and detect flavors.

    A directory is a flavor when one of its ancestors (within the dataset root)
    also contains a 'corpus' subdirectory — i.e. it extends an existing model
    rather than defining a new one.  Sub-model roots like name/citation or
    patent/citation are excluded because their parent dirs have no corpus.
    """
    dataset_dir = GROBID_ROOT / "grobid-trainer" / "resources" / "dataset"
    if not dataset_dir.exists():
        return {}

    result: Dict[str, List[str]] = {}

    for corpus_path in sorted(dataset_dir.rglob("corpus")):
        if not corpus_path.is_dir():
            continue
        candidate = corpus_path.parent          # the potential flavor/model dir
        rel_parts  = candidate.relative_to(dataset_dir).parts

        # Walk ancestors from the dataset root inward; stop at first with corpus.
        model_root: Optional[Path] = None
        for depth in range(len(rel_parts) - 1):
            ancestor = dataset_dir.joinpath(*rel_parts[: depth + 1])
            if (ancestor / "corpus").is_dir():
                model_root = ancestor
                break

        if model_root is not None:
            model_key  = str(model_root.relative_to(dataset_dir))
            flavor_str = str(candidate.relative_to(model_root))
            result.setdefault(model_key, []).append(flavor_str)

    return result


@app.get("/flavors", summary="List dataset flavors per model")
def list_flavors():
    """
    Return the flavors (dataset variants) that exist on disk for each model.

    A flavor is a subdirectory of a model's dataset directory that itself
    contains a corpus — e.g. header/article/light or segmentation/sdo/ietf.
    Sub-model paths such as name/citation are excluded.
    """
    return _discover_flavors()


# ── Upload registry ────────────────────────────────────────────────────────────

UPLOADS_DIR = GROBID_ROOT / "grobid-trainer" / "resources" / "uploads"
DATASET_DIR = GROBID_ROOT / "grobid-trainer" / "resources" / "dataset"


def _corpus_dir(model: str, flavor: str) -> Path:
    """Return the corpus directory for a model + flavor combination."""
    if flavor:
        return DATASET_DIR / model / flavor / "corpus"
    return DATASET_DIR / model / "corpus"


def _read_manifest(batch_id: str) -> Dict[str, Any]:
    manifest_path = UPLOADS_DIR / batch_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"Upload batch '{batch_id}' not found")
    with manifest_path.open() as f:
        return json.load(f)


@app.post("/upload/{model_name}", summary="Upload training data ZIP (non-destructive)")
async def upload_training_data(
    model_name: str,
    file: UploadFile = File(..., description="ZIP archive of training data files"),
    flavor: str = Form(default="", description="Model flavor, e.g. 'article/light'"),
    batch_name: str = Form(default="", description="Optional human-readable label for this batch"),
):
    """
    Upload a ZIP of training data files for a model (and optional flavor).

    Files are copied into the target corpus directory **only if they do not
    already exist** — existing files are never overwritten.  Every upload is
    recorded as a batch under `grobid-trainer/resources/uploads/{batch_id}/`
    so it can be inspected or reverted later.

    Files inside the ZIP are flattened: only the filename is used, any
    directory structure inside the ZIP is ignored.
    """
    corpus_dir = _corpus_dir(model_name, flavor)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    batch_id  = str(uuid.uuid4())[:8]
    batch_dir = UPLOADS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save the original archive
    zip_bytes = await file.read()
    (batch_dir / "archive.zip").write_bytes(zip_bytes)

    files_added: List[str]   = []
    files_skipped: List[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for entry in zf.infolist():
                if entry.is_dir():
                    continue
                # Flatten: use only the filename, drop any directory prefix
                filename = Path(entry.filename).name
                if not filename:
                    continue
                dest = corpus_dir / filename
                if dest.exists():
                    files_skipped.append(filename)
                else:
                    dest.write_bytes(zf.read(entry.filename))
                    files_added.append(filename)

    manifest = {
        "batch_id":      batch_id,
        "batch_name":    batch_name,
        "model":         model_name,
        "flavor":        flavor,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "corpus_dir":    str(corpus_dir),
        "files_added":   files_added,
        "files_skipped": files_skipped,
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "batch_id":           batch_id,
        "model":              model_name,
        "flavor":             flavor,
        "files_added_count":  len(files_added),
        "files_skipped_count": len(files_skipped),
        "files_added":        files_added,
        "files_skipped":      files_skipped,
    }


@app.get("/uploads", summary="List upload batches")
def list_uploads(model: str = "", flavor: str = ""):
    """
    List all recorded upload batches, optionally filtered by model and/or flavor.
    """
    if not UPLOADS_DIR.exists():
        return []
    result = []
    for batch_dir in sorted(UPLOADS_DIR.iterdir()):
        manifest_path = batch_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open() as f:
            m = json.load(f)
        if model  and m.get("model")  != model:
            continue
        if flavor and m.get("flavor") != flavor:
            continue
        result.append({
            "batch_id":           m["batch_id"],
            "batch_name":         m.get("batch_name", ""),
            "model":              m["model"],
            "flavor":             m.get("flavor", ""),
            "timestamp":          m["timestamp"],
            "files_added_count":  len(m.get("files_added", [])),
            "files_skipped_count": len(m.get("files_skipped", [])),
        })
    return result


@app.get("/uploads/{batch_id}", summary="Get upload batch details")
def get_upload(batch_id: str):
    """Return the full manifest for an upload batch."""
    return _read_manifest(batch_id)


@app.post("/revert/{batch_id}", summary="Revert an upload batch")
def revert_upload(batch_id: str):
    """
    Remove all files that were added by this upload batch from the corpus,
    then delete the batch record.  Files already removed are skipped (idempotent).
    """
    manifest   = _read_manifest(batch_id)
    corpus_dir = Path(manifest["corpus_dir"])
    removed    = []
    missing    = []

    for filename in manifest.get("files_added", []):
        target = corpus_dir / filename
        if target.exists():
            target.unlink()
            removed.append(filename)
        else:
            missing.append(filename)

    shutil.rmtree(UPLOADS_DIR / batch_id, ignore_errors=True)

    return {
        "batch_id":      batch_id,
        "files_removed": removed,
        "files_missing": missing,
    }


@app.get("/models/{model_name}", summary="List model files for a model")
def list_model_files(model_name: str, flavor: str = ""):
    """
    List all model files present in the model directory for a given model and
    optional flavor.

    Returns one entry per file matching `model.wapiti*`, including:
    - `name` — bare filename
    - `size_bytes` — file size
    - `modified_at` — ISO 8601 UTC modification timestamp
    - `is_active` — true if this is the currently active `model.wapiti`
    """
    if model_name not in VALID_MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'")
    mdir = _model_dir(model_name, flavor)
    if not mdir.exists():
        raise HTTPException(404, f"Model directory not found: {mdir}")
    entries = []
    for f in sorted(mdir.iterdir()):
        if f.name.startswith("model.wapiti") and f.is_file():
            stat = f.stat()
            entries.append({
                "name":        f.name,
                "size_bytes":  stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "is_active":   f.name == "model.wapiti",
            })
    return {"model": model_name, "flavor": flavor, "files": entries}


@app.delete("/models/{model_name}", summary="Delete a model file")
def delete_model_file(model_name: str, name: str, flavor: str = ""):
    """
    Delete a specific model file by name from the model directory.

    - `name` must be a bare filename (no path separators).
    - Deleting the active `model.wapiti` is not allowed (returns 409).
    - Returns 404 if the file does not exist.
    """
    if model_name not in VALID_MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'")
    if "/" in name or "\\" in name:
        raise HTTPException(400, "name must be a bare filename, not a path")
    if name == "model.wapiti":
        raise HTTPException(409, "Cannot delete the active model file 'model.wapiti'. "
                                 "Activate a different model first.")
    mdir   = _model_dir(model_name, flavor)
    target = mdir / name
    # Path traversal guard
    try:
        target.resolve().relative_to(mdir.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid filename")
    if not target.exists():
        raise HTTPException(404, f"Model file '{name}' not found")
    target.unlink()
    return {"model": model_name, "flavor": flavor, "deleted": name}


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
