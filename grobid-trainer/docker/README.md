# GROBID Trainer Service — Docker images

Three Dockerfiles are provided for different deployment targets.
All produce an image that runs [`trainer_service.py`](../../grobid-home/scripts/trainer_service.py)
on port **8072** and optionally dials out to a
[WebSocket relay server](../../doc/training-service.md#remote-operation-via-websocket-tunnel-hpc--apptainer)
for control from outside a closed HPC network.

## Variants

| Dockerfile | DeLFT / TF / JEP | GPU | ~Size | Use case |
|---|---|---|---|---|
| `Dockerfile.trainer.crf` | No | No | ~1 GB | Local CRF/Wapiti training |
| `Dockerfile.trainer.hpc` | Yes | via `--nv` | ~7 GB | HPC: CRF + DeLFT training |
| `Dockerfile.trainer.hpc.crf` | No | No | ~1 GB | HPC: CRF/Wapiti training only |

**Build context is always the project root**, not this directory.
The `-f` flag points to the Dockerfile; all `COPY` paths are relative to the root.

## Building

Copy the committed template and fill in your credentials (`.env` itself is gitignored):

```bash
cp grobid-trainer/docker/.env.template grobid-trainer/docker/.env
# then edit .env with your Docker Hub username, token, and preferred variant
```

Then run from anywhere in the project:

```bash
bash grobid-trainer/docker/build-and-push-image.sh
```

Uses **podman** if available, falls back to **docker buildx**.
Cross-compilation to `linux/amd64` works on Apple Silicon via QEMU (slow but correct).

### Manual build

```bash
# from the project root
podman build \
  -f grobid-trainer/docker/Dockerfile.trainer.hpc.crf \
  --build-arg GROBID_VERSION=0.8.3-SNAPSHOT \
  -t myuser/grobid-trainer:latest \
  .
```

## Running locally

```bash
# No tunnel — access directly on port 8072
docker run --rm -p 8072:8072 myuser/grobid-trainer:latest

# With WebSocket relay tunnel
docker run --rm -p 8072:8072 \
  -e RELAY_URL=ws://relay-host:8080/tunnel \
  -e RELAY_TOKEN=mysecret \
  myuser/grobid-trainer:latest

# Persist training data and models
docker run --rm -p 8072:8072 \
  -v /host/dataset:/opt/grobid/grobid-trainer/resources/dataset \
  -v /host/models:/opt/grobid/grobid-home/models \
  myuser/grobid-trainer:latest
```

## Running on HPC with Apptainer

### Pull the image (login node, once)

```bash
apptainer pull grobid-trainer.sif docker://myuser/grobid-trainer:latest
```

### CRF-only job (`Dockerfile.trainer.hpc.crf`)

No GPU needed — Wapiti is CPU-only.

```bash
#!/bin/bash
#SBATCH --job-name=grobid-train-crf
#SBATCH --time=12:00:00
#SBATCH --mem=16G

apptainer run \
  --bind /scratch/$USER/dataset:/opt/grobid/grobid-trainer/resources/dataset \
  --bind /scratch/$USER/models:/opt/grobid/grobid-home/models \
  --env RELAY_URL=ws://relay.your-domain.com:8080/tunnel \
  --env RELAY_TOKEN="$RELAY_TOKEN" \
  grobid-trainer.sif
```

### DeLFT job (`Dockerfile.trainer.hpc`)

The image contains **CPU-only TensorFlow**. GPU access is provided at runtime
by Apptainer's `--nv` flag, which injects the host NVIDIA driver and CUDA
libraries into the container — no CUDA wheels are baked into the image
(saves ~5 GB vs `tensorflow[and-cuda]`).

DeLFT embeddings (GloVe etc.) are not baked in either; bind-mount them from
scratch storage to avoid re-downloading on every job.

```bash
#!/bin/bash
#SBATCH --job-name=grobid-train-delft
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

apptainer run \
  --nv \
  --bind /scratch/$USER/dataset:/opt/grobid/grobid-trainer/resources/dataset \
  --bind /scratch/$USER/models:/opt/grobid/grobid-home/models \
  --bind /scratch/$USER/embeddings:/opt/grobid/grobid-home/resources/embeddings \
  --env RELAY_URL=ws://relay.your-domain.com:8080/tunnel \
  --env RELAY_TOKEN="$RELAY_TOKEN" \
  grobid-trainer-hpc.sif
```

## Environment variables

| Variable | Description |
|---|---|
| `RELAY_URL` | WebSocket URL of the relay server, e.g. `ws://host:8080/tunnel` |
| `RELAY_TOKEN` | Shared secret for relay authentication |
| `SERVICE_PORT` | HTTP port for the trainer service (default: `8072`) |

## Path layout inside the container

| Path | Contents |
|---|---|
| `/opt/grobid/grobid-home/` | Native libs, config, models |
| `/opt/grobid/grobid-trainer/resources/dataset/` | Training data — bind-mount here |
| `/opt/grobid/grobid-trainer/resources/uploads/` | Upload batch records |
| `/opt/grobid/grobid-home/models/` | Trained models — bind-mount here |
| `/opt/grobid/grobid-home/resources/embeddings/` | DeLFT embeddings — bind-mount here |
