"""Run replay viewer in a separate process."""

import asyncio
import threading
import webbrowser
from multiprocessing import Queue

import src_py.replay_viewer as replay_viewer


def run_viewer(queue: Queue) -> None:
    """Start servers and forward queue messages to clients."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    threading.Thread(
        target=replay_viewer.start_http_server,
        daemon=True,
    ).start()

    async def queue_consumer() -> None:
        """Consume queue and send to WebSocket clients."""
        while True:
            export_string, recording_string, frames = await loop.run_in_executor(
                None, queue.get
            )

            await replay_viewer.send_data(
                export_string,
                recording_string,
                frames,
            )

    async def main() -> None:
        """Run queue consumer and WebSocket server."""
        asyncio.create_task(queue_consumer())
        await replay_viewer.start_ws_server()

    webbrowser.open("http://localhost:8000")

    loop.run_until_complete(main())
