"""
Simple HTTP + WebSocket server for serving a site and sending frame data.
"""

import asyncio
import threading
import http.server
import webbrowser
import json
from typing import Set

import websockets
from websockets.asyncio.server import ServerConnection

HTTP_PORT: int = 8000
WS_PORT: int = 8765
DIRECTORY: str = "replay_viewer"

clients: Set[ServerConnection] = set()
loop: asyncio.AbstractEventLoop | None = None


class Handler(http.server.SimpleHTTPRequestHandler):
    """Fast static file handler."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        return

    def address_string(self) -> str:
        return self.client_address[0]


def start_http_server() -> None:
    """Start the HTTP server."""
    with http.server.ThreadingHTTPServer(("", HTTP_PORT), Handler) as httpd:
        print(f"HTTP server running at http://localhost:{HTTP_PORT}")
        httpd.serve_forever()


async def ws_handler(websocket: ServerConnection) -> None:
    """Handle a WebSocket connection."""
    clients.add(websocket)
    print("Client connected")

    try:
        async for message in websocket:
            print("Received:", message)
    finally:
        clients.remove(websocket)
        print("Client disconnected")


async def start_ws_server() -> None:
    """Start the WebSocket server."""
    async with websockets.serve(ws_handler, "localhost", WS_PORT):
        print(f"WebSocket server running at ws://localhost:{WS_PORT}")
        await asyncio.Future()


async def send_data(export_string: str, recording_string: str, frames: int) -> None:
    """Send data to all connected clients."""
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


def send(export_string: str, recording_string: str, frames: int) -> None:
    """Thread-safe send."""
    if loop is None:
        return

    asyncio.run_coroutine_threadsafe(
        send_data(export_string, recording_string, frames),
        loop,
    )


def start() -> None:
    """Start servers and open the browser."""

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )
    http_thread.start()

    def run_loop():
        global loop  # pylint: disable=global-statement
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_ws_server())

    ws_thread = threading.Thread(target=run_loop, daemon=True)
    ws_thread.start()

    webbrowser.open(f"http://localhost:{HTTP_PORT}")
