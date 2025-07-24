from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import time
import mimetypes
import torch
import threading

from config import config, Args
from util import pil_to_frame, bytes_to_pil, is_firefox
from connection_manager import ConnectionManager, ServerFullException
from vid2vid import Pipeline

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120
# logging.basicConfig(level=logging.DEBUG)


class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.produce_predictions_stop_event = None
        self.produce_predictions_task = None
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id, self.pipeline)
                if self.produce_predictions_stop_event is not None:
                    self.produce_predictions_stop_event.set()
                if self.produce_predictions_task is not None:
                    self.produce_predictions_task.cancel()
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            last_frame_time = None
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id, self.pipeline)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    if data["status"] != "next_frame":
                        await asyncio.sleep(THROTTLE)
                        continue

                    params = await self.conn_manager.receive_json(user_id)
                    params = self.pipeline.InputParams(**params)
                    info = self.pipeline.Info()
                    params = SimpleNamespace(**params.dict())
                    if info.input_mode == "image":
                        image_data = await self.conn_manager.receive_bytes(user_id)
                        if len(image_data) == 0:
                            await self.conn_manager.send_json(
                                user_id, {"status": "send_frame"}
                            )
                            await asyncio.sleep(THROTTLE)
                            continue
                        params.image = bytes_to_pil(image_data)
                    await self.conn_manager.update_data(user_id, params)
                    await self.conn_manager.send_json(user_id, {"status": "wait"})
                    if last_frame_time is None:
                        last_frame_time = time.time()
                    else:
                        # print(f"Frame time: {time.time() - last_frame_time}")
                        last_frame_time = time.time()

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id, self.pipeline)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                async def push_frames_to_pipeline():
                    last_params = SimpleNamespace()
                    while True:
                        params = await self.conn_manager.get_latest_data(user_id)
                        if vars(params) and params.__dict__ != last_params.__dict__:
                            last_params = params
                            self.pipeline.accept_new_params(params)
                        await self.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )

                async def generate():
                    MIN_FPS = 5
                    MAX_FPS = 30
                    SMOOTHING = 0.8  # EMA smoothing factor

                    last_burst_time = time.time()
                    last_queue_size = 0
                    sleep_time = 1 / 20  # Initial guess
                    last_frame_time = None

                    # Initialize moving average frame interval
                    ema_frame_interval = sleep_time
                    while True:
                        queue_size = await self.conn_manager.get_output_queue_size(user_id)
                        print(f"Queue size: {queue_size}")
                        if queue_size > last_queue_size:
                            # A new burst has come in
                            current_burst_time = time.time()
                            elapsed = current_burst_time - last_burst_time

                            if queue_size > 0 and elapsed > 0:
                                raw_interval = elapsed / queue_size
                                ema_frame_interval = SMOOTHING * ema_frame_interval + (1 - SMOOTHING) * raw_interval
                                sleep_time = min(max(ema_frame_interval, 1 / MAX_FPS), 1 / MIN_FPS)

                            last_burst_time = current_burst_time

                        last_queue_size = queue_size
                        try:
                            frame = await self.conn_manager.get_frame(user_id)
                            if frame is None:
                                break
                            yield frame
                            if not is_firefox(request.headers["user-agent"]):
                                yield frame
                            if last_frame_time is None:
                                last_frame_time = time.time()
                            else:
                                # print(f"Frame time: {time.time() - last_frame_time}")
                                last_frame_time = time.time()
                        except Exception as e:
                            print(f"Frame fetch error: {e}")
                            break

                        await asyncio.sleep(sleep_time)

                def produce_predictions(user_id, loop, stop_event):
                    while not stop_event.is_set():
                        images = self.pipeline.predict()
                        if len(images) == 0:
                            time.sleep(THROTTLE)
                            continue
                        asyncio.run_coroutine_threadsafe(
                            self.conn_manager.put_frames_to_output_queue(
                                user_id,
                                list(map(pil_to_frame, images))
                            ),
                            loop
                        )

                self.produce_predictions_stop_event = threading.Event()
                self.produce_predictions_task = asyncio.create_task(asyncio.to_thread(
                    produce_predictions, user_id, asyncio.get_running_loop(), self.produce_predictions_stop_event
                ))
                asyncio.create_task(push_frames_to_pipeline())
                await self.conn_manager.send_json(user_id, {"status": "send_frame"})

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )

            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                # Stop prediction thread on error
                if self.produce_predictions_stop_event is not None:
                    self.produce_predictions_stop_event.set()
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = self.pipeline.Info.schema()
            info = self.pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = self.pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        if not os.path.exists("./frontend/public"):
            os.makedirs("./frontend/public")

        self.app.mount(
            "/", StaticFiles(directory="./frontend/public", html=True), name="public"
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(config, device)
    return _pipeline

app = App(config, get_pipeline()).app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=False,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )
