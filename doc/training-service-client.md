# GROBID Trainer Service CLI Client

`grobid-trainer/scripts/training-api-client.py` is a command-line client for the [GROBID Trainer HTTP Service](training-service.md).  It requires only Python 3 and the standard library — no third-party packages, no Nix environment.

## Usage

```bash
python grobid-trainer/scripts/training-api-client.py [--host HOST] [--port PORT] <command> [options]
```

**Global options:**

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--host` | `localhost` | Service hostname or IP |
| `--port` | `8072` | Service port |

---

## Commands

### `health` — Check service status

```bash
python training-api-client.py health
```

Prints the full JSON response from `GET /health`, including platform info, JAR location, Java library path, and the list of available models.

---

### `models` — List available models

```bash
python training-api-client.py models
```

Prints one model name per line (suitable for piping or `grep`).  The list comes from the `models` field of `/health` and reflects the contents of `grobid-home/models/`.

---

### `flavors` — List dataset flavors

```bash
python training-api-client.py flavors [model]
```

Lists flavors discovered from the dataset directory.  Without `model`, prints all flavors as tab-separated `model<TAB>flavor` pairs.  With a model name, prints only that model's flavors, one per line.

**Examples:**

```bash
# All flavors
python training-api-client.py flavors

# Flavors for segmentation only
python training-api-client.py flavors segmentation
```

---

### `train` — Start model training

```bash
python training-api-client.py train <model> [options]
```

Submits a training job and streams the log live until the job completes.

**Options:**

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--mode MODE` | `0` | `0`=train only, `1`=evaluate only, `2`=auto-split train+eval, `3`=N-fold cross-eval |
| `--seg-ratio RATIO` | `0.8` | Train/eval split ratio (mode 2 only) |
| `--n-folds N` | `10` | Number of folds (mode 3 only) |
| `--incremental` | off | Start from existing model instead of training from scratch |
| `--flavor FLAVOR` | `""` | Model flavor, e.g. `article/light` |
| `--no-stream` | off | Submit job and return immediately without streaming the log |

**Examples:**

```bash
# Train the date model using all corpus data
python training-api-client.py train date

# Split corpus 80/20, train and evaluate
python training-api-client.py train citation --mode 2

# 10-fold cross-evaluation
python training-api-client.py train header --mode 3

# Incremental training of a flavored model
python training-api-client.py train segmentation --incremental --flavor article/dh-law-footnotes

# Submit without waiting for output
python training-api-client.py train date --no-stream
```

The command exits with code `0` on success, non-zero if training fails.

---

### `eval` — Run end-to-end evaluation

```bash
python training-api-client.py eval <nlm|tei> --p2t <path> [options]
```

Submits an end-to-end evaluation job against a gold-standard dataset and streams the output.

**Options:**

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--p2t PATH` | required | Absolute path to the gold-standard dataset directory |
| `--run` | off | Run GROBID on PDFs first (requires GROBID service on port 8070) |
| `--file-ratio RATIO` | `1.0` | Fraction of dataset files to use (0.0–1.0) |
| `--flavor FLAVOR` | `""` | Model flavor |
| `--no-stream` | off | Submit job and return immediately |

**Examples:**

```bash
# Evaluate against pre-processed NLM gold set
python training-api-client.py eval nlm --p2t /data/PMC_sample_1943

# Process PDFs then evaluate using 10% of the dataset
python training-api-client.py eval nlm --p2t /data/PMC_sample_1943 --run --file-ratio 0.1
```

---

### `jobs` — List all jobs

```bash
python training-api-client.py jobs
```

Prints a table of all submitted jobs with their IDs, status, start time, model/flavor (or eval type/flavor for evaluation jobs), and a truncated command.

---

### `status` — Get job status

```bash
python training-api-client.py status <job_id> [-v]
```

Prints a one-line summary of the job (status, model/flavor, duration, exit code).  With `-v` / `--verbose`, also prints the full JSON response and the complete log.

---

### `stop` — Stop a running job

```bash
python training-api-client.py stop <job_id>
```

Sends SIGTERM to the job's subprocess and marks it as `cancelled`.  Returns immediately — does not wait for the process to exit.  The partial log remains accessible via `status <job_id>`.

Returns an error if the job is not currently running.

**Example:**

```bash
python training-api-client.py stop a3f1bc7e
# → job_id=a3f1bc7e  status=cancelled
```

---

### `stream` — Stream a job's log

```bash
python training-api-client.py stream <job_id>
```

Connects to the SSE stream for a job and prints log lines live.  Useful for attaching to a job that was submitted with `--no-stream`.

---

### `upload` — Upload training data

```bash
python training-api-client.py upload <model> <directory> [options]
```

Zips the contents of `<directory>` and uploads them to the service.  Files that already exist in the target corpus are skipped — existing data is never overwritten.  Returns a `batch_id` that can be used to revert the upload.

**Options:**

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--flavor FLAVOR` | `""` | Target flavor, e.g. `article/dh-law-footnotes` |
| `--batch-name NAME` | `""` | Human-readable label for this batch |
| `-v` / `--verbose` | off | List each added and skipped file |

**Examples:**

```bash
# Upload to the default corpus
python training-api-client.py upload header ./my-header-data

# Upload to a specific flavor, with a label
python training-api-client.py upload segmentation ./footnote-data \
  --flavor article/dh-law-footnotes \
  --batch-name "DH law footnotes v1" \
  -v
```

Output:

```text
Zipped /path/to/footnote-data → 284,672 bytes
batch_id=c11fb1c3  files_added=42  files_skipped=0
  Added:
    doc001.train
    doc002.train
    ...
```

!!! note
    Directory contents are flattened: any subdirectory structure inside `<directory>` is ignored and only the bare filenames are placed in the corpus.

---

### `uploads` — List upload batches

```bash
python training-api-client.py uploads [--model MODEL] [--flavor FLAVOR]
```

Prints a table of all recorded upload batches.  Use `--model` and/or `--flavor` to filter.

**Examples:**

```bash
# All batches
python training-api-client.py uploads

# Only segmentation batches
python training-api-client.py uploads --model segmentation

# A specific flavor
python training-api-client.py uploads --model segmentation --flavor article/dh-law-footnotes
```

Output:

```text
BATCH ID  MODEL         FLAVOR                    TIMESTAMP            ADDED  SKIPPED  NAME
---------------------------------------------------------------------------------------------------
c11fb1c3  segmentation  article/dh-law-footnotes  2026-03-04T16:48:33     42        0  DH law footnotes v1
```

---

### `revert` — Revert an upload batch

```bash
python training-api-client.py revert <batch_id> [-v]
```

Removes all files that were **added** by the given batch from the corpus, then deletes the batch record.  Files that were skipped at upload time are not affected.  The operation is safe to run even if some files have already been removed manually.

With `-v` / `--verbose`, lists each removed and missing file.

**Example:**

```bash
python training-api-client.py revert c11fb1c3 -v
```

Output:

```text
batch_id=c11fb1c3  files_removed=42  files_missing=0
  removed: doc001.train
  removed: doc002.train
  ...
```

!!! warning
    Reverting a batch is permanent. The batch record is deleted and cannot be recovered. The original ZIP is also removed. Upload the same directory again if you need to re-add the files.

---

## Typical workflow

```bash
# 1. Check the service is running and the JAR is built
python training-api-client.py health

# 2. See what models and flavors are available
python training-api-client.py models
python training-api-client.py flavors

# 3. Upload new training data for a flavor
python training-api-client.py upload segmentation ./my-data \
  --flavor article/dh-law-footnotes \
  --batch-name "initial load" -v

# 4. Train the model with the new data (mode 2: auto-split train+eval)
python training-api-client.py train segmentation \
  --flavor article/dh-law-footnotes \
  --mode 2

# 5. If results are unsatisfactory, revert the uploaded data
python training-api-client.py revert <batch_id>
```
