# GROBID Trainer HTTP Service for Apple Silicon

!!! important "Platform-specific notes"
    The service described here is experimental and has been tested only on Apple Silicon, specifically on a M4-Mac (32GB), with a dependency on
    the [`nix` environment](use-nix-for-apple-silicon.md). In principle,
    is should also work on other platforms but your mileage may vary.

## Goal

The trainer service wraps GROBID model training and end-to-end evaluation in a REST API. Instead of running Gradle tasks directly from the command line, you can trigger and monitor training jobs over HTTP — useful for scripting, CI workflows, or remote operation.

The service is implemented in [`grobid-home/scripts/trainer_service.py`](../grobid-home/scripts/trainer_service.py) and runs on port **8072** by default.

!!! important "Prerequisites"
    The trainer service must be run inside the Nix development environment to have access to Java 21, the correct native libraries, and (optionally) the Metal GPU for DeLFT models. See [Using the Nix environment](use-nix-environment.md) for setup instructions.

## Quick start

```bash
# 1. Enter the Nix environment
nix develop

# 2. Build the trainer JAR (once, or after source changes)
./gradlew :grobid-trainer:shadowJar --no-daemon

# 3. Start the service
python grobid-home/scripts/trainer_service.py

# 4. Verify it is running
curl http://localhost:8072/health
```

Interactive API documentation (Swagger UI) is available at `http://localhost:8072/docs` while the service is running.

## How it works

The service is a lightweight [FastAPI](https://fastapi.tiangolo.com/) application. When a training request arrives, it:

1. Locates the pre-built `grobid-trainer-*-onejar.jar`
2. Constructs the correct `java.library.path` for the current platform (macOS ARM64 → `grobid-home/lib/mac_arm-64`; JEP location detected from `VIRTUAL_ENV`)
3. Launches the trainer as a subprocess in the background
4. Streams stdout/stderr into an in-memory log buffer
5. Returns a `job_id` immediately so the caller can poll or stream the output

Multiple jobs can run concurrently, though training is CPU/memory-intensive — running more than one heavy model at a time on a single machine is not recommended.

## Training a model

```bash
POST /train/{model_name}
```

**Path parameter:** `model_name` — one of the model names listed in the [Valid models](#valid-models) section.

**Request body (JSON):**

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `mode` | int | `0` | Training mode (see table below) |
| `seg_ratio` | float | `0.8` | Train/eval split ratio used in mode 2 |
| `n_folds` | int | `10` | Number of folds used in mode 3 |
| `incremental` | bool | `false` | Start from existing model instead of training from scratch |
| `flavor` | string | `""` | Model flavour variant (see [Flavours](#flavours)) |

**Training modes:**

| Mode | Description | Data used |
| ---- | ----------- | --------- |
| `0` | Train only | All files in `grobid-trainer/resources/dataset/{model}/corpus/` |
| `1` | Evaluate only | All files in `grobid-trainer/resources/dataset/{model}/evaluation/` |
| `2` | Auto-split, then train + evaluate | Corpus split by `seg_ratio`; no files needed in `evaluation/` |
| `3` | N-fold cross-evaluation | All corpus files, rotated across `n_folds` folds |

**Example — train the `date` model from scratch:**

```bash
curl -X POST http://localhost:8072/train/date \
  -H "Content-Type: application/json" \
  -d '{"mode": 0}'
```

**Example — split corpus 90/10, train and evaluate:**

```bash
curl -X POST http://localhost:8072/train/citation \
  -H "Content-Type: application/json" \
  -d '{"mode": 2, "seg_ratio": 0.9}'
```

**Example — 10-fold cross-evaluation of the header model:**

```bash
curl -X POST http://localhost:8072/train/header \
  -H "Content-Type: application/json" \
  -d '{"mode": 3, "n_folds": 10}'
```

**Example — incremental training (continue from existing model):**

```bash
curl -X POST http://localhost:8072/train/citation \
  -H "Content-Type: application/json" \
  -d '{"mode": 0, "incremental": true}'
```

**Response:**

```json
{
  "job_id": "a3f1bc7e",
  "status": "running",
  "model": "date",
  "mode": 0
}
```

Trained models are written to `grobid-home/models/{model}/` and replace the previous model. A backup is saved as `model.wapiti.old`.

## End-to-end evaluation

End-to-end evaluation measures GROBID's full extraction pipeline (PDF → structured output) against a gold-standard dataset. See [End-to-end evaluation](End-to-end-evaluation.md) for dataset preparation details.

```bash
POST /evaluate/{eval_type}
```

**Path parameter:** `eval_type` — `nlm` (JATS/NLM format) or `tei` (TEI format).

**Request body (JSON):**

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `p2t` | string | required | Absolute path to the gold-standard dataset directory |
| `run` | bool | `false` | Run GROBID on the PDFs before evaluating (requires GROBID service on port 8070) |
| `file_ratio` | float | `1.0` | Fraction of the dataset to use (0.0–1.0) |
| `flavor` | string | `""` | Model flavour variant |

!!! tip "run=false vs run=true"
    - `run=false` (default): evaluate GROBID output files already present in the dataset directory. Use this to re-score without reprocessing.
    - `run=true`: call the running GROBID service (port 8070) to process PDFs, then evaluate. Start the GROBID service first.

**Example — evaluate against PubMedCentral gold set (PDFs already processed):**

```bash
curl -X POST http://localhost:8072/evaluate/nlm \
  -H "Content-Type: application/json" \
  -d '{"p2t": "/data/PMC_sample_1943", "run": false}'
```

**Example — full run: process PDFs then evaluate on 10% of the dataset:**

```bash
curl -X POST http://localhost:8072/evaluate/nlm \
  -H "Content-Type: application/json" \
  -d '{"p2t": "/data/PMC_sample_1943", "run": true, "file_ratio": 0.1}'
```

The evaluation report is written to `grobid-home/tmp/report.md`.

## Monitoring jobs

### Get job status and full log

```bash
GET /jobs/{job_id}
```

```bash
curl http://localhost:8072/jobs/a3f1bc7e
```

Response:

```json
{
  "job_id": "a3f1bc7e",
  "status": "done",
  "exit_code": 0,
  "log": "Loading model...\n[training output...]\nDone.",
  "start_time": "2024-03-01T10:00:00",
  "end_time":   "2024-03-01T10:12:34",
  "duration_s": 754.1,
  "pid": 12345,
  "cmd": "java -Xmx4g ... grobid-trainer-0.8.3-onejar.jar 0 date -gH ..."
}
```

Possible `status` values: `running`, `done`, `failed`.

### Stream live output (Server-Sent Events)

```bash
GET /jobs/{job_id}/stream
```

```bash
curl -N http://localhost:8072/jobs/a3f1bc7e/stream
```

Each log line is delivered as an SSE `data` event. When the job finishes, a final `event: done` message is sent with the exit code:

```text
data: Loading Wapiti model...
data: Iteration 1/100 — loss: 0.342
data: Iteration 2/100 — loss: 0.298
...
event: done
data: exit_code=0
```

### List all jobs

```bash
GET /jobs
```

```bash
curl http://localhost:8072/jobs
```

## Health check

```bash
GET /health
```

```bash
curl http://localhost:8072/health
```

Example response:

```json
{
  "status": "ok",
  "platform": "Darwin/arm64",
  "grobid_root": "/Users/you/grobid",
  "grobid_home": "/Users/you/grobid/grobid-home",
  "jar": "/Users/you/grobid/grobid-trainer/build/libs/grobid-trainer-0.8.3-SNAPSHOT-onejar.jar",
  "jar_built": true,
  "java_lib_path": "/Users/you/grobid/grobid-home/lib/mac_arm-64:/Users/you/grobid/.venv/lib/python3.11/site-packages/jep",
  "virtual_env": "/Users/you/grobid/.venv",
  "models": ["affiliation-address", "citation", "date", "fulltext", "header", ...]
}
```

Check `jar_built: true` before submitting training jobs. If `false`, build the JAR first:

```bash
./gradlew :grobid-trainer:shadowJar --no-daemon
```

## Valid models

The following model names are accepted by `POST /train/{model_name}`:

| Model name | Description |
| ---------- | ----------- |
| `date` | Date parsing |
| `header` | Header metadata extraction |
| `name-header` | Author names in header |
| `name-citation` | Author names in citations |
| `affiliation-address` | Affiliation and address parsing |
| `citation` | Bibliographic reference parsing |
| `segmentation` | Document segmentation |
| `fulltext` | Full-text structuring (CRF only) |
| `reference-segmentation` | Reference list segmentation |
| `figure` | Figure detection |
| `table` | Table detection |
| `patent-citation` | Patent citation parsing |
| `funding-acknowledgement` | Funding and acknowledgement extraction |
| `shorttext` | Short-text classification |
| `ebook-model` | E-book processing |

## Flavours

Some models support variants that are trained on specific document types. Pass the flavour string in the `flavor` field:

| Flavour | Applicable models | Description |
| ------- | ----------------- | ----------- |
| `""` (empty) | all | Default general model |
| `article/light` | `header`, `segmentation`, `fulltext` | Lightweight article model |
| `article/light-ref` | `header`, `segmentation`, `fulltext` | Lightweight article model with references |
| `sdo/ietf` | `header`, `segmentation` | IETF standards documents |

## Command-line options

```bash
python grobid-home/scripts/trainer_service.py --help

usage: trainer_service.py [-h] [--host HOST] [--port PORT]

optional arguments:
  --host HOST   Bind address (default: 0.0.0.0)
  --port PORT   Bind port (default: 8072)
```

## Relation to Gradle training tasks

The service invokes the pre-built `grobid-trainer-*-onejar.jar` directly rather than going through Gradle. This avoids Gradle startup overhead for each request. The JAR's main class (`org.grobid.trainer.TrainerRunner`) accepts the same arguments that the Gradle tasks pass internally:

```text
TrainerRunner  <mode>  <model_name>  -gH <grobid_home>  [-s <seg_ratio>]  [-n <n_folds>]  [-i]
```

End-to-end evaluation uses a second class in the same JAR:

```text
EndToEndEvaluation  <nlm|tei>  <p2t>  <run>  <file_ratio>  <flavor>
```

Both classes are equivalent to running the corresponding Gradle tasks (`train_*`, `jatsEval`, `teiEval`).

## Troubleshooting

### `jar_built: false` in `/health`

The trainer JAR has not been built yet:

```bash
./gradlew :grobid-trainer:shadowJar --no-daemon
```

### Job status is `failed` with exit code `1`

Check the full log for details:

```bash
curl http://localhost:8072/jobs/{job_id}
```

Common causes: missing training data in `grobid-trainer/resources/dataset/{model}/corpus/`, insufficient heap space (increase `-Xmx` in the service source), or a DeLFT import error if using deep-learning models.

### Deep-learning model fails with `jep.JepException`

The JEP library or DeLFT is not correctly installed. Verify:

```bash
# Inside nix develop
python -c "import jep; import delft; print('ok')"
```

If this fails, rebuild the venv:

```bash
rm -rf .venv && nix develop
```

### Training takes unexpectedly long

CRF (Wapiti) training is CPU-bound and benefits from the M4's efficiency cores. DeLFT training uses the Metal GPU via `tensorflow-metal`; verify GPU is active:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

For CRF, you can increase parallelism by raising `nbThreads` in `grobid-home/config/grobid.yaml`.
