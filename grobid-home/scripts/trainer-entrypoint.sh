#!/bin/sh
# Docker entrypoint for the GROBID trainer service.
# Reads optional RELAY_URL and RELAY_TOKEN environment variables and passes
# them as CLI args to trainer_service.py.
#
# Environment variables:
#   RELAY_URL    WebSocket URL of the relay server, e.g. ws://relay-host:8080/tunnel
#   RELAY_TOKEN  Shared secret for relay authentication
#   SERVICE_PORT Bind port for the HTTP service (default: 8072)

set -e

PORT="${SERVICE_PORT:-8072}"

set -- python3 /opt/grobid/grobid-home/scripts/trainer_service.py \
    --host 0.0.0.0 \
    --port "$PORT"

if [ -n "${RELAY_URL:-}" ]; then
    set -- "$@" --relay "$RELAY_URL"
fi

if [ -n "${RELAY_TOKEN:-}" ]; then
    set -- "$@" --relay-token "$RELAY_TOKEN"
fi

exec "$@"
