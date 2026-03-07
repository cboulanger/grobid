#!/usr/bin/env bash
# Build and push a GROBID trainer service image to Docker Hub.
# Uses podman if available, falls back to docker buildx.
# Edit .env (next to this script) to select the Dockerfile variant and
# set your Docker Hub credentials before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # project root (two levels up)
ENV_FILE="$SCRIPT_DIR/.env"

# ── Load .env ─────────────────────────────────────────────────────────────────

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found."
    echo "Copy the template block from README.md into grobid-trainer/docker/.env."
    exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

# ── Validate required vars ────────────────────────────────────────────────────

missing=()
for var in DOCKERHUB_USERNAME DOCKERHUB_TOKEN IMAGE_NAME DOCKERFILE GROBID_VERSION; do
    [ -z "${!var:-}" ] && missing+=("$var")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "Error: the following variables are not set in .env: ${missing[*]}"
    exit 1
fi

DOCKERFILE_PATH="$SCRIPT_DIR/$DOCKERFILE"
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE_PATH"
    echo "Valid values for DOCKERFILE:"
    echo "  Dockerfile.trainer.crf"
    echo "  Dockerfile.trainer.hpc"
    echo "  Dockerfile.trainer.hpc.crf"
    exit 1
fi

IMAGE_TAG="${IMAGE_TAG:-$GROBID_VERSION}"
FULL_IMAGE="$DOCKERHUB_USERNAME/$IMAGE_NAME:$IMAGE_TAG"
PLATFORM="${BUILD_PLATFORM:-linux/amd64}"

# ── Select container tool ─────────────────────────────────────────────────────

if command -v podman &>/dev/null; then
    TOOL=podman
    echo "==> Using podman"
elif command -v docker &>/dev/null; then
    TOOL=docker
    echo "==> Using docker"
else
    echo "Error: neither podman nor docker found in PATH"
    exit 1
fi

# ── Login ─────────────────────────────────────────────────────────────────────

echo "==> Logging in to Docker Hub as $DOCKERHUB_USERNAME …"
if [ "$TOOL" = podman ]; then
    echo "$DOCKERHUB_TOKEN" | podman login docker.io -u "$DOCKERHUB_USERNAME" --password-stdin
else
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi

# ── Build & push ──────────────────────────────────────────────────────────────

echo ""
echo "==> Building $FULL_IMAGE"
echo "    Dockerfile : $DOCKERFILE"
echo "    Platform   : $PLATFORM"
echo "    Context    : $ROOT_DIR"
echo ""

if [ "$TOOL" = podman ]; then
    podman build \
        --platform "$PLATFORM" \
        --build-arg "GROBID_VERSION=$GROBID_VERSION" \
        --file "$DOCKERFILE_PATH" \
        --tag "$FULL_IMAGE" \
        "$ROOT_DIR"

    echo ""
    echo "==> Pushing $FULL_IMAGE …"
    podman push "$FULL_IMAGE"
else
    docker buildx build \
        --platform "$PLATFORM" \
        --build-arg "GROBID_VERSION=$GROBID_VERSION" \
        --file "$DOCKERFILE_PATH" \
        --tag "$FULL_IMAGE" \
        --push \
        "$ROOT_DIR"
fi

echo ""
echo "==> Done.  Image available at:"
echo "    docker pull $FULL_IMAGE"
echo ""
echo "    Apptainer (HPC):"
echo "    apptainer pull docker://$FULL_IMAGE"
