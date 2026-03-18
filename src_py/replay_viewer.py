"""HTTP + WebSocket server for replay viewing."""

import asyncio
import http.server
import json
from typing import Set

import websockets
from websockets.asyncio.server import ServerConnection

HTTP_PORT: int = 8000
WS_PORT: int = 8765
DIRECTORY: str = "replay_viewer"

clients: Set[ServerConnection] = set()


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve static files without logging."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        """Initialize handler with fixed directory."""
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    # pylint: disable=redefined-builtin
    def log_message(self, format, *args):
        """Suppress request logs."""
        return

    def address_string(self) -> str:
        """Return client IP."""
        return self.client_address[0]


def start_http_server() -> None:
    """Run the HTTP server."""
    with http.server.ThreadingHTTPServer(("", HTTP_PORT), Handler) as httpd:
        print(f"HTTP server running at http://localhost:{HTTP_PORT}")
        httpd.serve_forever()


async def ws_handler(websocket: ServerConnection) -> None:
    """Handle a WebSocket client."""
    clients.add(websocket)
    print("Client connected")

    try:
        async for message in websocket:
            print("Received:", message)
    finally:
        clients.remove(websocket)
        print("Client disconnected")


async def start_ws_server() -> None:
    """Run the WebSocket server forever."""
    async with websockets.serve(ws_handler, "localhost", WS_PORT):
        print(f"WebSocket server running at ws://localhost:{WS_PORT}")
        await asyncio.Future()


async def send_data(export_string: str, recording_string: str, frames: int) -> None:
    """Send replay data to all clients."""
    if not clients:
        return

    payload = {
        "exportString": export_string,
        "recordingString": recording_string,
        "frames": frames,
    }

    msg = json.dumps(payload)

    await asyncio.gather(
        *(client.send(msg) for client in clients),
        return_exceptions=True,
    )
