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

## Training and evaluating models

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
| `epsilon` | float | `null` | Wapiti convergence threshold for this job only — overrides `grobid.yaml`. Omit to use the configured default. |
| `nb_max_iterations` | int | `null` | Maximum Wapiti iterations for this job only — overrides `grobid.yaml`. Omit to use the configured default. |
| `keep_existing` | bool | `false` | If `true`, the new model is saved as `model.wapiti.{timestamp}` and the current active model is preserved. If `false` (default), the new model becomes active and the previous is saved as `model.wapiti.{timestamp}`. |
| `model_file` | string | `null` | **Mode 1 only.** Specific model filename to evaluate, e.g. `model.wapiti.20260305-143022`. Omit to use the active `model.wapiti`. |

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

**Example — fast development run (loose epsilon, capped iterations):**

```bash
curl -X POST http://localhost:8072/train/header \
  -H "Content-Type: application/json" \
  -d '{"mode": 2, "epsilon": 0.0001, "nb_max_iterations": 500}'
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

Trained models are written to `grobid-home/models/{model}/`. After training completes, the previous model is saved as `model.wapiti.{timestamp}` (e.g. `model.wapiti.20260305-143022`). The `trained_model_file` field in the job record reports the filename of the resulting model.

When `keep_existing=true`, the newly trained model is saved with a timestamp instead, and the existing active `model.wapiti` is left unchanged. Use this to test a new model without replacing the production one. Use `GET /models/{model_name}` to list all available versions and `DELETE /models/{model_name}` to remove one.

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
  "model": "date",
  "flavor": "",
  "mode": 0,
  "keep_existing": true,
  "log": "Loading model...\n[training output...]\nDone.",
  "start_time": "2024-03-01T10:00:00",
  "end_time":   "2024-03-01T10:12:34",
  "duration_s": 754.1,
  "pid": 12345,
  "cmd": "java -Xmx4g ... 0 date -gH /path/to/grobid-home -epsilon 0.001 -maxIter 200 -modelPath /path/to/grobid-home/models/date/model.wapiti.20240301-101234",
  "trained_model_file": "model.wapiti.20240301-101234"
}
```

`trained_model_file` is set only for completed training jobs (modes 0, 2, 3); it is `null` for evaluation jobs and while the job is still running.

- `keep_existing=false` (default): `"model.wapiti"` — the new model replaced the active one
- `keep_existing=true`: `"model.wapiti.{timestamp}"` — the new model was saved with a timestamp; the active model is unchanged

Possible `status` values: `running`, `done`, `failed`, `cancelled`.

### Stop a running job

```bash
POST /jobs/{job_id}/stop
```

```bash
curl -X POST http://localhost:8072/jobs/a3f1bc7e/stop
```

Sends SIGTERM to the job's subprocess and immediately marks the job as `cancelled`.  The job record and its partial log are preserved and can still be retrieved via `GET /jobs/{job_id}`.

Response:

```json
{ "job_id": "a3f1bc7e", "status": "cancelled" }
```

Returns **409** if the job is not currently running (already `done`, `failed`, or `cancelled`).

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

Returns a summary list of all jobs.  Each entry includes `model` (training jobs) or `eval_type` (evaluation jobs), `flavor`, and `trained_model_file` (set for completed training jobs; `null` otherwise).

```bash
curl http://localhost:8072/jobs
```

### Delete a job record

```bash
DELETE /jobs/{job_id}
```

Removes a finished (`done`, `failed`, or `cancelled`) job record from memory.  Returns **409** if the job is still running, **404** if not found.

```bash
curl -X DELETE http://localhost:8072/jobs/a3f1bc7e
```

Response:

```json
{ "deleted": "a3f1bc7e" }
```

### Delete all non-running job records

```bash
DELETE /jobs
```

Removes all job records that are not currently `running` in a single call.  Running jobs are silently skipped.

```bash
curl -X DELETE http://localhost:8072/jobs
```

Response:

```json
{ "deleted": ["a3f1bc7e", "b4c2de8f"], "skipped_running": 1 }
```

## Managing model files

### List model files

```bash
GET /models/{model_name}[?flavor=<flavor>]
```

Lists all `model.wapiti*` files present in the model directory, with size, modification time, and whether it is the currently active model.

```bash
curl "http://localhost:8072/models/header?flavor=article/light"
```

Response:

```json
{
  "model": "header",
  "flavor": "article/light",
  "files": [
    {
      "name": "model.wapiti",
      "size_bytes": 18432714,
      "modified_at": "2026-03-05T14:30:22+00:00",
      "is_active": true
    },
    {
      "name": "model.wapiti.20260305-143022",
      "size_bytes": 17891230,
      "modified_at": "2026-03-05T12:15:01+00:00",
      "is_active": false
    }
  ]
}
```

Returns **404** if the model directory does not exist.

### Delete a model file

```bash
DELETE /models/{model_name}?name=<filename>[&flavor=<flavor>]
```

Deletes a specific model file by name.  The `name` parameter must be a bare filename (no path separators).  Deleting the active `model.wapiti` is not allowed (returns **409**).

```bash
curl -X DELETE "http://localhost:8072/models/header?name=model.wapiti.20260305-143022&flavor=article/light"
```

Response:

```json
{ "deleted": "model.wapiti.20260305-143022" }
```

Returns **400** if `name` contains path separators, **404** if the file does not exist, **409** if you attempt to delete the active `model.wapiti`.

---

## Listing available flavors

```bash
GET /flavors
```

Returns a dictionary mapping each model name (as it appears in the dataset directory) to the list of flavor paths that have a corpus on disk.  Only models that actually have flavored datasets are included; models with only the default corpus are omitted.

```bash
curl http://localhost:8072/flavors
```

Example response:

```json
{
  "fulltext":     ["article/light"],
  "header":       ["article/light", "sdo/ietf"],
  "segmentation": ["article/dh-law-footnotes", "article/light", "sdo/ietf"]
}
```

The flavor strings returned here can be passed directly as the `flavor` field in training requests.

## Uploading training data

```bash
POST /upload/{model_name}
```

Uploads a ZIP archive of training data files for a model and optional flavor.  Files are copied into the correct corpus directory **non-destructively**: if a file with the same name already exists it is skipped, never overwritten.  Every upload is recorded as a *batch* so it can be inspected or reverted later.

**Multipart form fields:**

| Field | Required | Description |
| ----- | -------- | ----------- |
| `file` | yes | ZIP archive containing training data files |
| `flavor` | no | Model flavor, e.g. `article/dh-law-footnotes` (default: `""`) |
| `batch_name` | no | Optional human-readable label for this batch |

Files inside the ZIP are **flattened**: any directory structure is ignored and only the bare filename is used.  This matches the flat layout expected in the corpus directories.

If the target corpus directory does not exist (e.g. for a new flavor) it is created automatically.

**Example — upload data for a new segmentation flavor:**

```bash
curl -X POST http://localhost:8072/upload/segmentation \
  -F "file=@/path/to/my-data.zip" \
  -F "flavor=article/dh-law-footnotes" \
  -F "batch_name=initial dataset"
```

**Response:**

```json
{
  "batch_id":            "c11fb1c3",
  "model":               "segmentation",
  "flavor":              "article/dh-law-footnotes",
  "files_added_count":   42,
  "files_skipped_count": 0,
  "files_added":         ["doc001.train", "doc002.train", "..."],
  "files_skipped":       []
}
```

Record the `batch_id` if you may need to revert this upload later.

### Batch storage

Each batch is saved under `grobid-trainer/resources/uploads/{batch_id}/` (gitignored):

```text
uploads/
  c11fb1c3/
    manifest.json   ← model, flavor, timestamp, files_added, files_skipped
    archive.zip     ← original ZIP kept for reference
```

### Listing upload batches

```bash
GET /uploads[?model=<model>&flavor=<flavor>]
```

Returns a summary list of all recorded batches, optionally filtered by model and/or flavor.

```bash
curl "http://localhost:8072/uploads?model=segmentation"
```

### Inspecting a batch

```bash
GET /uploads/{batch_id}
```

Returns the full manifest for a single batch, including the complete file lists.

```bash
curl http://localhost:8072/uploads/c11fb1c3
```

### Reverting an upload

```bash
POST /revert/{batch_id}
```

Removes every file that was **added** by this batch from the corpus, then deletes the batch record.  Files that were skipped at upload time (already existed) are not touched.  If a file has already been removed manually, it is skipped silently — the operation is idempotent within a single batch.

```bash
curl -X POST http://localhost:8072/revert/c11fb1c3
```

**Response:**

```json
{
  "batch_id":      "c11fb1c3",
  "files_removed": ["doc001.train", "doc002.train", "..."],
  "files_missing": []
}
```

Once reverted the batch record is deleted; a subsequent call to the same `batch_id` returns 404.

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
                          [--relay RELAY] [--relay-token RELAY_TOKEN]

optional arguments:
  --host HOST               Bind address (default: 0.0.0.0)
  --port PORT               Bind port (default: 8072)
  --relay RELAY             Relay WS URL to dial out to, e.g. wss://host/tunnel
  --relay-token RELAY_TOKEN Shared secret for relay auth (or set RELAY_TOKEN env var)
```

## Relation to Gradle training tasks

The service invokes the pre-built `grobid-trainer-*-onejar.jar` directly rather than going through Gradle. This avoids Gradle startup overhead for each request. The JAR's main class (`org.grobid.trainer.TrainerRunner`) accepts the following arguments:

```text
TrainerRunner  <mode>  <model_name>  -gH <grobid_home>
               [-s <seg_ratio>]  [-n <n_folds>]  [-i]
               [-epsilon <float>]  [-maxIter <int>]
               [-modelPath <absolute_path>]
```

`-epsilon` and `-maxIter` override the Wapiti training parameters per-job without touching `grobid.yaml`. `-modelPath` sets the output path for training (mode 0) or the input model path for evaluation (mode 1).

End-to-end evaluation uses a second class in the same JAR:

```text
EndToEndEvaluation  <nlm|tei>  <p2t>  <run>  <file_ratio>  <flavor>
```

Both classes are equivalent to running the corresponding Gradle tasks (`train_*`, `jatsEval`, `teiEval`).

## Tuning training speed (epsilon)

CRF (Wapiti) training time is dominated by the L-BFGS convergence threshold `epsilon`. The default values in `grobid.yaml` are tuned for production accuracy and can make training take many hours. The `epsilon` and `nb_max_iterations` fields let you override these per-request without editing any config file.

| Preset | `epsilon` | `nb_max_iterations` | Use case |
| ------ | --------- | ------------------- | -------- |
| **Fast** | `0.001` | `200` | Smoke-test: verify the pipeline runs end-to-end in minutes |
| **Development** | `0.0001` | `500` | Iterate on training data: good enough accuracy, finishes in under an hour |
| **Production** | *(omit — use default)* | *(omit)* | Final model: uses the tight threshold from `grobid.yaml` |

A looser epsilon stops the optimizer earlier at the cost of a small accuracy drop (typically < 1 F1 point). For iterative data development, **Development** is the recommended preset: it converges fast enough to give meaningful feedback while still producing a usable model.

The override is applied only for the submitted job. `grobid.yaml` is never modified.

## Smoke test

A temporary end-to-end smoke test script is provided at [`grobid-trainer/scripts/smoke_test.py`](../grobid-trainer/scripts/smoke_test.py).  It exercises the full model-versioning workflow against a running service:

1. Health check (verifies JAR is built and the model exists)
2. Train with fast settings (`epsilon=0.001`, `nb_max_iterations=200`) and `keep_existing=true`
3. Verify the timestamped model file appears in `GET /models/{model}`
4. Evaluate (mode 1) using the timestamped model file and print the field-level and instance-level metrics
5. Delete the timestamped file via `DELETE /models/{model}`
6. Verify the file is gone and the active `model.wapiti` is intact

```bash
# Default: model=date, no flavor
python grobid-trainer/scripts/smoke_test.py

# Override model and/or flavor
python grobid-trainer/scripts/smoke_test.py --model header --flavor article/light

# Point at a remote service
python grobid-trainer/scripts/smoke_test.py --host 192.168.1.10 --port 8072
```

The script exits `0` on success and `1` with a descriptive error on any failure.  Training at `epsilon=0.001` / 200 iterations typically completes in a few minutes for most models.

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

---

## Remote operation via WebSocket tunnel (HPC / Apptainer)

HPC clusters are typically not reachable from the internet — you cannot open a port into the job node. The WebSocket tunnel lets the container **dial out** to a relay server you control, which then proxies your HTTP requests (including live SSE streams) through that connection.

```text
[Your laptop / CI]         [Relay server — internet-accessible]         [HPC job node]

  HTTP client ──────────► relay_server.py (port 8080) ◄── WS tunnel ── trainer_service.py
  curl, browser, scripts       proxies requests inward                   (port 8072, local)
```

Three files are involved:

| File | Runs on | Purpose |
| ---- | ------- | ------- |
| `grobid-home/scripts/relay_server.py` | Relay server | Accepts the tunnel; proxies HTTP through it |
| `grobid-home/scripts/tunnel_client.py` | HPC container | Dials out to relay; forwards requests to local service |
| `grobid-home/scripts/trainer_service.py` | HPC container | Existing trainer service; starts tunnel client if `--relay` is set |

### 1. Start the relay server

On any internet-accessible host (VPS, your workstation behind a port-forward, etc.):

```bash
pip install fastapi "uvicorn[standard]" websockets
RELAY_TOKEN=mysecret python grobid-home/scripts/relay_server.py --port 8080
```

Or with explicit flags:

```bash
python grobid-home/scripts/relay_server.py --host 0.0.0.0 --port 8080 --token mysecret
```

The relay exposes:

| Endpoint | Description |
| -------- | ----------- |
| `WS /tunnel` | Tunnel connection from the HPC container |
| `GET /tunnel/status` | Check whether a container is currently connected |
| `ANY /{path}` | Proxied to the trainer service through the tunnel |

### 1b. Put the relay behind nginx (recommended)

If you are running the relay on a host that already serves other content via nginx, expose it on a path prefix rather than a raw port. Add these two location blocks inside your `server` block, **before** the catch-all `location /`:

```nginx
# WebSocket tunnel — HPC container dials in here
location /tunnel {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}

# REST + SSE endpoints — clients access the trainer through here
location /relay/ {
    proxy_pass http://127.0.0.1:8080/;   # trailing slash strips the prefix
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;                  # required for SSE streaming
    proxy_cache off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    client_max_body_size 500M;            # for large training data uploads
}
```

Apply the config:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

After this, the HPC container connects to `wss://your-domain/tunnel` and all trainer API requests go to `https://your-domain/relay/`.

!!! note "Why two location blocks?"
    The WebSocket `/tunnel` endpoint needs HTTP/1.1 upgrade headers and a very long read timeout (the connection stays open for the entire training job). The `/relay/` prefix handles regular HTTP and SSE traffic; `proxy_buffering off` is essential here so SSE chunks are not held in nginx's buffer before being forwarded to the client.

### 2. Start the trainer service with tunnel enabled

Pass `--relay` (and optionally `--relay-token`) when starting the service inside the HPC container.  The tunnel client is launched automatically as a background task:

```bash
python grobid-home/scripts/trainer_service.py \
    --relay wss://relay.your-domain.com/tunnel \
    --relay-token mysecret
```

`RELAY_TOKEN` can be used instead of `--relay-token`:

```bash
export RELAY_TOKEN=mysecret
python grobid-home/scripts/trainer_service.py --relay wss://relay.your-domain.com/tunnel
```

Once connected, all endpoints described in this document are accessible through the relay URL, including `GET /jobs/{job_id}/stream` (SSE streams correctly end-to-end).

### 3. Use the relay as if it were the local service

If running the relay **directly** (no nginx, raw port 8080):

```bash
# Health check via relay
curl http://relay.your-domain.com:8080/health

# Upload training data
curl -X POST http://relay.your-domain.com:8080/upload/segmentation \
  -F "file=@my-data.zip" -F "flavor=article/dh-law-footnotes"

# Start training
curl -X POST http://relay.your-domain.com:8080/train/header \
  -H "Content-Type: application/json" \
  -d '{"mode": 2, "epsilon": 0.0001}'

# Stream live output (SSE — works transparently through the tunnel)
curl -N http://relay.your-domain.com:8080/jobs/a3f1bc7e/stream
```

If running **behind nginx** with the `/relay/` prefix (see [step 1b](#1b-put-the-relay-behind-nginx-recommended)):

```bash
# Health check via relay
curl https://your-domain/relay/health

# Upload training data
curl -X POST https://your-domain/relay/upload/segmentation \
  -F "file=@my-data.zip" -F "flavor=article/dh-law-footnotes"

# Start training
curl -X POST https://your-domain/relay/train/header \
  -H "Content-Type: application/json" \
  -d '{"mode": 2, "epsilon": 0.0001}'

# Stream live output (SSE — works transparently through the tunnel)
curl -N https://your-domain/relay/jobs/a3f1bc7e/stream
```

### Running as an Apptainer / Slurm job

Pull the image once on the HPC login node (internet access required):

```bash
apptainer pull grobid-trainer.sif docker://you/grobid-trainer:latest
```

Paths inside the container follow the standard GROBID layout under `/opt/grobid/`.  Bind-mount your scratch directories there so training data and trained models survive across job runs:

```bash
#!/bin/bash
#SBATCH --job-name=grobid-train
#SBATCH --time=12:00:00
#SBATCH --mem=16G

apptainer run \
  --nv \
  --bind /scratch/$USER/dataset:/opt/grobid/grobid-trainer/resources/dataset \
  --bind /scratch/$USER/models:/opt/grobid/grobid-home/models \
  --env RELAY_URL=ws://relay.your-domain.com:8080/tunnel \
  --env RELAY_TOKEN="$RELAY_TOKEN" \
  grobid-trainer.sif
```

The container needs no inbound ports.  `RELAY_URL` and `RELAY_TOKEN` are read by the container entrypoint and forwarded to `trainer_service.py` automatically.

### Tunnel CLI options

**`trainer_service.py`** (HPC container side):

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--relay URL` | `""` | Relay WS URL. If omitted, no tunnel is started. |
| `--relay-token SECRET` | `$RELAY_TOKEN` | Shared secret for authentication. |

**`relay_server.py`** (relay side):

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--host ADDR` | `0.0.0.0` | Bind address. |
| `--port PORT` | `8080` | Bind port. |
| `--token SECRET` | `$RELAY_TOKEN` | If set, tunnel connections without a matching token are rejected. |

### Running the tunnel client standalone

`tunnel_client.py` can also be run independently of the trainer service — useful for testing connectivity or wrapping a service started by other means:

```bash
python grobid-home/scripts/tunnel_client.py \
    --relay wss://relay.your-domain.com/tunnel \
    --token mysecret \
    --service http://localhost:8072
```

The client reconnects automatically with exponential back-off (up to 60 s) if the relay is temporarily unreachable.

### Security notes

- Set a strong `RELAY_TOKEN`.  Without it the tunnel endpoint is unauthenticated and anyone can connect.
- Run the relay behind TLS (`wss://` not `ws://`) — use a reverse proxy such as nginx or Caddy, or pass `--ssl-certfile` / `--ssl-keyfile` to uvicorn directly.
- Only one tunnel connection is accepted at a time.  A second container connecting with the correct token will replace the first.
- The relay does not restrict which paths are forwarded.  Access control is left to the caller (network firewall, VPN, etc.).
