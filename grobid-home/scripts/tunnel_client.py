#!/usr/bin/env python3
"""
GROBID Trainer Tunnel Client
==============================
Runs inside the HPC container alongside trainer_service.py.  Dials out to the
relay server and forwards HTTP requests (including SSE streams) to the local
trainer service.

Usage (standalone):
    python tunnel_client.py --relay wss://relay-host/tunnel --token SECRET

Or imported and started as an asyncio task from trainer_service.py:
    from tunnel_client import connect
    asyncio.create_task(connect("wss://...", token="..."))

Protocol (JSON frames over WebSocket):

  Inbound  (relay → client):
    { "id": "...", "method": "GET", "path": "/jobs/abc/stream",
      "query": "", "headers": {...}, "body_b64": "" }

  Outbound, non-streaming:
    { "id": "...", "type": "response",
      "status": 200, "headers": {...}, "body_b64": "..." }

  Outbound, SSE/streaming:
    { "id": "...", "type": "response_start", "status": 200, "headers": {...} }
    { "id": "...", "type": "response_chunk", "chunk_b64": "..." }  (repeated)
    { "id": "...", "type": "response_end" }
"""

import asyncio
import base64
import json
import logging
import os
from typing import Optional

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("tunnel_client")

# Hop-by-hop headers that must not be forwarded to the local service
_HOP_BY_HOP = frozenset({
    "host", "content-length", "transfer-encoding", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "upgrade",
})


async def _handle_request(
    ws,
    envelope: dict,
    service_url: str,
    send_lock: asyncio.Lock,
) -> None:
    """
    Forward one HTTP request envelope to the local trainer service and send
    the response (or stream of response chunks) back over the WebSocket.

    Runs as an independent asyncio Task so multiple requests can be
    in-flight simultaneously without blocking the receive loop.
    """
    req_id  = envelope["id"]
    method  = envelope["method"]
    path    = envelope["path"]
    query   = envelope.get("query", "")
    headers = {
        k: v for k, v in envelope.get("headers", {}).items()
        if k.lower() not in _HOP_BY_HOP
    }
    body = base64.b64decode(envelope.get("body_b64", ""))
    url  = service_url.rstrip("/") + path + (f"?{query}" if query else "")

    async def send(msg: dict) -> None:
        async with send_lock:
            await ws.send(json.dumps(msg))

    try:
        # Use httpx streaming for all requests so we can detect SSE before
        # reading the body, then decide whether to stream or buffer.
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(method, url, headers=headers, content=body) as resp:
                resp_headers = dict(resp.headers)
                content_type = resp_headers.get("content-type", "")

                if "text/event-stream" in content_type:
                    # ── Streaming (SSE) ───────────────────────────────────────
                    await send({
                        "id":      req_id,
                        "type":    "response_start",
                        "status":  resp.status_code,
                        "headers": resp_headers,
                    })
                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if chunk:
                            await send({
                                "id":        req_id,
                                "type":      "response_chunk",
                                "chunk_b64": base64.b64encode(chunk).decode(),
                            })
                    await send({"id": req_id, "type": "response_end"})

                else:
                    # ── Buffered ──────────────────────────────────────────────
                    # aread() consumes and buffers the entire body; safe because
                    # all non-SSE responses are small JSON payloads.
                    await resp.aread()
                    await send({
                        "id":       req_id,
                        "type":     "response",
                        "status":   resp.status_code,
                        "headers":  resp_headers,
                        "body_b64": base64.b64encode(resp.content).decode(),
                    })

    except Exception as exc:
        log.error(f"[tunnel] request {req_id} ({method} {path}) failed: {exc}")
        # Send a synthetic 502 so the relay caller gets a response instead of timing out
        try:
            await send({
                "id":       req_id,
                "type":     "response",
                "status":   502,
                "headers":  {"content-type": "application/json"},
                "body_b64": base64.b64encode(
                    json.dumps({"detail": f"Tunnel error forwarding request: {exc}"}).encode()
                ).decode(),
            })
        except Exception:
            pass  # WebSocket may already be dead; the relay will time out that request


async def connect(
    relay_url: str,
    *,
    token: str = "",
    service_url: str = "http://localhost:8072",
    initial_delay: float = 5.0,
    max_delay: float = 60.0,
) -> None:
    """
    Connect to the relay server and forward requests indefinitely.

    Reconnects automatically on disconnect with exponential back-off.
    Designed to run as a long-lived asyncio Task (never returns normally).

    Args:
        relay_url:     WebSocket URL of the relay server, e.g. wss://host/tunnel
        token:         Shared secret sent as ?token= query parameter
        service_url:   Base URL of the local trainer service
        initial_delay: Seconds to wait before the first reconnect attempt
        max_delay:     Maximum back-off delay in seconds
    """
    ws_url = relay_url
    if token:
        sep    = "&" if "?" in ws_url else "?"
        ws_url = f"{ws_url}{sep}token={token}"

    delay = initial_delay
    while True:
        try:
            log.info(f"[tunnel] Connecting to {relay_url} …")
            async with websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
                # Allow large frames for file uploads
                max_size=256 * 1024 * 1024,
            ) as ws:
                log.info("[tunnel] Connected — ready to forward requests")
                delay = initial_delay   # reset back-off on successful connect

                send_lock = asyncio.Lock()
                async for raw in ws:
                    envelope: dict = json.loads(raw)
                    # Dispatch each request as an independent task so slow
                    # requests (training, SSE streams) don't block newer ones.
                    asyncio.create_task(
                        _handle_request(ws, envelope, service_url, send_lock)
                    )

        except ConnectionClosed as exc:
            log.warning(f"[tunnel] Disconnected ({exc}). Reconnecting in {delay}s …")
        except OSError as exc:
            msg = str(exc)
            if "record layer failure" in msg or "SSL" in msg:
                log.warning(
                    f"[tunnel] SSL handshake failed — relay is probably plain HTTP. "
                    f"Use ws:// instead of wss:// (or add TLS to the relay). "
                    f"Retrying in {delay}s …"
                )
            else:
                log.warning(f"[tunnel] Could not connect: {exc}. Retrying in {delay}s …")
        except Exception as exc:
            log.exception(f"[tunnel] Unexpected error: {exc}. Retrying in {delay}s …")

        await asyncio.sleep(delay)
        delay = min(delay * 2, max_delay)


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="GROBID Trainer Tunnel Client")
    parser.add_argument("--relay",   required=True,                              help="Relay WS URL, e.g. wss://host/tunnel")
    parser.add_argument("--token",   default=os.environ.get("RELAY_TOKEN", ""), help="Shared secret for tunnel auth")
    parser.add_argument("--service", default="http://localhost:8072",            help="Local trainer service URL")
    args = parser.parse_args()

    asyncio.run(connect(args.relay, token=args.token, service_url=args.service))
