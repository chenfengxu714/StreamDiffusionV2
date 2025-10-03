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
import threading
import multiprocessing as mp
import signal
import sys

from config import config, Args
from util import pil_to_frame, bytes_to_pil, is_firefox
from connection_manager import ConnectionManager, ServerFullException

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
        self.shutdown_event = asyncio.Event()
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
                # Do not block shutdown here; schedule disconnect
                asyncio.create_task(self.conn_manager.disconnect(user_id, self.pipeline))
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
                    # Refresh idle timer on any client control message
                    last_time = time.time()
                    # Handle stop/pause without closing socket: go idle and wait
                    if data and data.get("status") == "stop":
                        await self.conn_manager.send_json(user_id, {"status": "wait"})
                        await asyncio.sleep(THROTTLE)
                        continue
                    # Mark upload completion: after this, don't receive image bytes again
                    if data and data.get("status") == "upload_done":
                        self.conn_manager.set_video_upload_completed(user_id, True)
                        print(f"[Main] Upload completed for user {user_id}")
                        await self.conn_manager.send_json(user_id, {"status": "upload_done_ack"})
                        continue
                    if not data or data.get("status") != "next_frame":
                        await asyncio.sleep(THROTTLE)
                        continue

                    params = await self.conn_manager.receive_json(user_id)
                    params = self.pipeline.InputParams(**params)
                    info = self.pipeline.Info()
                    params = SimpleNamespace(**params.dict())
                    
                    # Check if upload mode is enabled
                    is_upload_mode = params.__dict__.get('input_mode') == 'upload' or params.__dict__.get('upload_mode', False)
                    self.conn_manager.set_upload_mode(user_id, is_upload_mode)
                    if is_upload_mode:
                        print(f"[Main] Upload mode detected for user {user_id}")
                    
                    if info.input_mode == "image":
                        upload_completed = self.conn_manager.is_video_upload_completed(user_id)
                        # Only receive image bytes if not in upload mode, or upload not completed yet
                        if (not is_upload_mode) or (is_upload_mode and not upload_completed):
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                # await asyncio.sleep(sleep_time)
                                continue
                            # If upload mode and not completed, append frames to cache for later reuse
                            if is_upload_mode and not upload_completed:
                                await self.conn_manager.add_video_frame(user_id, image_data)
                                print(f"[Main] Added frame to video queue for user {user_id}")
                            # For camera mode, set current image directly
                            if not is_upload_mode:
                                params.image = bytes_to_pil(image_data)
                        else:
                            # Upload already completed: do not receive more bytes; image will be fed from cached frames
                            pass
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
                    sleep_time = 1 / 20  # Initial guess
                    while True:
                        # Check if upload mode is enabled
                        video_status = self.conn_manager.get_video_queue_status(user_id)
                        is_upload_mode = video_status.get("is_upload_mode", False)
                        
                        if is_upload_mode:
                            # Upload mode: get next frame from video queue
                            video_frame = await self.conn_manager.get_next_video_frame(user_id)
                            if video_frame:
                                # Create params object with video frame
                                params = SimpleNamespace()
                                params.image = bytes_to_pil(video_frame)
                                # Copy other parameters
                                if vars(last_params):
                                    for key, value in last_params.__dict__.items():
                                        if key != 'image':
                                            setattr(params, key, value)
                                
                                if params.__dict__ != last_params.__dict__:
                                    last_params = params
                                    self.pipeline.accept_new_params(params)
                                    print(f"[Main] Upload mode: sent frame to pipeline for user {user_id}")
                                # Yield control without delaying to maximize fluency
                                # await asyncio.sleep(sleep_time)
                            else:
                                # No frame available, wait a bit
                                await asyncio.sleep(sleep_time)
                        else:
                            # Camera mode: normal processing
                            params = await self.conn_manager.get_latest_data(user_id)
                            if vars(params) and params.__dict__ != last_params.__dict__:
                                last_params = params
                                self.pipeline.accept_new_params(params)
                            await self.conn_manager.send_json(
                                user_id, {"status": "send_frame"}
                            )
                            # Yield control without delaying
                            # await asyncio.sleep(sleep_time)

                async def generate():
                    MIN_FPS = 5
                    MAX_FPS = 30
                    SMOOTHING = 0.8  # EMA smoothing factor

                    last_burst_time = time.time()
                    last_queue_size = 0
                    sleep_time = 1 / 20  # Initial guess
                    last_frame_time = None
                    frame_time_list = []

                    # Initialize moving average frame interval
                    ema_frame_interval = sleep_time
                    while True:
                        queue_size = await self.conn_manager.get_output_queue_size(user_id)
                        if queue_size > last_queue_size:
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
                                frame_time_list.append(time.time() - last_frame_time)
                                if len(frame_time_list) > 100:
                                    frame_time_list.pop(0)
                                last_frame_time = time.time()
                        except Exception as e:
                            print(f"Frame fetch error: {e}")
                            break

                        await asyncio.sleep(sleep_time)

                def produce_predictions(user_id, loop, stop_event):
                    while not stop_event.is_set():
                        images = self.pipeline.produce_outputs()
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

        # Add shutdown event handler
        @self.app.on_event("shutdown")
        async def shutdown_event():
            print("[App] Shutdown event triggered, cleaning up...")
            await self.cleanup()

    async def cleanup(self):
        """Clean up all resources on shutdown"""
        print("[App] Starting cleanup process...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Stop all background tasks
        if self.produce_predictions_stop_event is not None:
            self.produce_predictions_stop_event.set()
            print("[App] Stopped prediction tasks")
        
        if self.produce_predictions_task is not None:
            self.produce_predictions_task.cancel()
            try:
                await self.produce_predictions_task
            except asyncio.CancelledError:
                pass
            print("[App] Cancelled prediction task")
        
        # Close all WebSocket connections and pipeline
        print(f"[App] Closing {len(self.conn_manager.active_connections)} active connections...")
        try:
            await self.conn_manager.disconnect_all(self.pipeline)
        except Exception as e:
            print(f"[App] Error during disconnect_all: {e}")
        
        print("[App] Cleanup completed")


# Global app instance for signal handler
app_instance = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n[Main] Received signal {signum}, shutting down gracefully...")
    if app_instance:
        # Trigger cleanup in a separate thread to avoid blocking
        import threading
        def trigger_cleanup():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app_instance.cleanup())
                loop.close()
            except Exception as e:
                print(f"[Main] Error during cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=trigger_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        cleanup_thread.join(timeout=5)  # Wait up to 5 seconds for cleanup
    
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mp.set_start_method("spawn", force=True)

    config.pretty_print()
    if config.num_gpus > 1:
        from vid2vid_pipe import MultiGPUPipeline
        pipeline = MultiGPUPipeline(config)
    else:
        from vid2vid import Pipeline
        pipeline = Pipeline(config)

    app_obj = App(config, pipeline)
    app = app_obj.app
    app_instance = app_obj  # Set global reference for signal handler

    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=False,
            ssl_certfile=config.ssl_certfile,
            ssl_keyfile=config.ssl_keyfile,
        )
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received, shutting down...")
        # Trigger cleanup
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(app_obj.cleanup())
            loop.close()
        except Exception as e:
            print(f"[Main] Error during cleanup: {e}")
        sys.exit(0)
    except Exception as e:
        print(f"[Main] Error: {e}")
        sys.exit(1)
