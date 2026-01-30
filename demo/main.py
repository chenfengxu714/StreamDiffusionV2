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
from collections import deque

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
        # Initialize metrics collection only if enabled
        self.enable_metrics = config.enable_metrics
        self.target_latency = config.target_latency  # Target latency in seconds for deadline miss rate
        self.step = config.step  # Pipeline step parameter
        self.gpu_ids = config.gpu_ids  # GPU IDs (e.g., "0,1" or "0")
        if self.enable_metrics:
            # Simple timestamp queue for input frames (FIFO)
            self.user_input_timestamps = {}  # user_id -> deque of input timestamps
            self.user_metrics_lock = threading.Lock()  # Lock for thread-safe timestamp tracking
            # Track metrics collection count per user (number of batches collected)
            self.user_batch_count = {}  # user_id -> count of batches collected
            self.user_latency_history = {}  # user_id -> list of latencies for statistics
            self.user_raw_data = {}  # user_id -> list of raw batch data (for logging)
            self.metrics_log_dir = "./slo_metrics"
            os.makedirs(self.metrics_log_dir, exist_ok=True)
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
                # Clean up metrics and timestamp tracking for this user
                if self.enable_metrics:
                    with self.user_metrics_lock:
                        self.user_input_timestamps.pop(user_id, None)
                        self.user_batch_count.pop(user_id, None)
                        self.user_latency_history.pop(user_id, None)
                        self.user_raw_data.pop(user_id, None)
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
            # 16 FPS throttling: minimum interval between frames (1/16 seconds)
            TARGET_FPS = 16.0
            min_frame_interval = 1.0 / TARGET_FPS
            last_frame_received_time = None
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
                    if data and data.get("status") == "pause":
                        params = SimpleNamespace(**{"restart": True})
                        await self.conn_manager.update_data(user_id, params)
                        continue
                    if data and data.get("status") == "resume":
                        await self.conn_manager.send_json(user_id, {"status": "send_frame"})
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
                            
                            # 16 FPS throttling: only process frames at 16 FPS rate
                            current_time = time.time()
                            if last_frame_received_time is not None:
                                time_since_last_frame = current_time - last_frame_received_time
                                if time_since_last_frame < min_frame_interval:
                                    # Skip this frame to maintain 16 FPS
                                    await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                                    continue
                            
                            last_frame_received_time = current_time
                            
                            # If upload mode and not completed, append frames to cache for later reuse
                            if is_upload_mode and not upload_completed:
                                await self.conn_manager.add_video_frame(user_id, image_data)
                                print(f"[Main] Added frame to video queue for user {user_id} (16 FPS throttled)")
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
        
        @self.app.get("/api/metrics/{user_id}")
        async def get_metrics(user_id: uuid.UUID, window_size: int = 100):
            """Get SLO metrics for a specific user"""
            if not self.enable_metrics:
                return JSONResponse({"error": "Metrics collection is not enabled"}, status_code=400)
            try:
                import numpy as np
                with self.user_metrics_lock:
                    if user_id not in self.user_latency_history or len(self.user_latency_history[user_id]) == 0:
                        return JSONResponse({"error": "No metrics data available"}, status_code=404)
                    
                    latencies = np.array(self.user_latency_history[user_id][-window_size:])
                    
                    stats = {
                        "mean_latency": float(np.mean(latencies)),
                        "median_latency": float(np.median(latencies)),
                        "p95_latency": float(np.percentile(latencies, 95)),
                        "p99_latency": float(np.percentile(latencies, 99)),
                        "min_latency": float(np.min(latencies)),
                        "max_latency": float(np.max(latencies)),
                        "std_latency": float(np.std(latencies)),
                        "sample_count": len(latencies),
                        "remaining_frames": len(self.user_input_timestamps.get(user_id, deque())),
                        "batch_count": self.user_batch_count.get(user_id, 0)
                    }
                    
                    return JSONResponse(stats)
            except Exception as e:
                logging.error(f"Error getting metrics: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                async def push_frames_to_pipeline():
                    last_params = SimpleNamespace()
                    sleep_time = 1 / 20  # Initial guess
                    # 16 FPS throttling for upload mode
                    TARGET_FPS = 16.0
                    min_frame_interval = 1.0 / TARGET_FPS
                    last_frame_sent_time = None
                    while True:
                        # Check if upload mode is enabled
                        video_status = self.conn_manager.get_video_queue_status(user_id)
                        is_upload_mode = video_status.get("is_upload_mode", False)
                        
                        if is_upload_mode:
                            # Upload mode: get next frame from video queue with 16 FPS throttling
                            current_time = time.time()
                            if last_frame_sent_time is not None:
                                time_since_last_frame = current_time - last_frame_sent_time
                                if time_since_last_frame < min_frame_interval:
                                    # Wait to maintain 16 FPS
                                    await asyncio.sleep(min_frame_interval - time_since_last_frame)
                            
                            video_frame = await self.conn_manager.get_next_video_frame(user_id)
                            if video_frame:
                                last_frame_sent_time = time.time()
                                # Create params object with video frame
                                params = SimpleNamespace()
                                params.image = bytes_to_pil(video_frame)
                                # Copy other parameters
                                if vars(last_params):
                                    for key, value in last_params.__dict__.items():
                                        if key != 'image' and key != '_frame_id':
                                            setattr(params, key, value)
                                
                                if params.__dict__ != last_params.__dict__:
                                    # Record input timestamp when frame is added to pipeline queue
                                    if self.enable_metrics:
                                        input_timestamp = time.time()
                                        with self.user_metrics_lock:
                                            if user_id not in self.user_input_timestamps:
                                                self.user_input_timestamps[user_id] = deque()
                                            self.user_input_timestamps[user_id].append(input_timestamp)
                                    
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
                                # Record input timestamp when frame is added to pipeline queue
                                if self.enable_metrics:
                                    input_timestamp = time.time()
                                    with self.user_metrics_lock:
                                        if user_id not in self.user_input_timestamps:
                                            self.user_input_timestamps[user_id] = deque()
                                        self.user_input_timestamps[user_id].append(input_timestamp)
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
                            
                            # Output timestamp is already recorded in produce_predictions
                            
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
                        
                        # Calculate latency for each output frame using FIFO timestamp queue
                        if self.enable_metrics:
                            output_timestamp = time.time()
                            batch_latencies = []
                            
                            with self.user_metrics_lock:
                                if user_id in self.user_input_timestamps:
                                    # For each output frame, get corresponding input timestamp (FIFO)
                                    for _ in range(len(images)):
                                        if len(self.user_input_timestamps[user_id]) > 0:
                                            input_timestamp = self.user_input_timestamps[user_id].popleft()
                                            latency = output_timestamp - input_timestamp
                                            batch_latencies.append(latency)
                                            
                                            # Add to history for statistics
                                            if user_id not in self.user_latency_history:
                                                self.user_latency_history[user_id] = []
                                            self.user_latency_history[user_id].append(latency)
                                    
                                    # Print batch statistics
                                    if len(batch_latencies) > 0:
                                        avg_latency = sum(batch_latencies) / len(batch_latencies)
                                        remaining_frames = len(self.user_input_timestamps[user_id])
                                        
                                        # Get batch count
                                        if user_id not in self.user_batch_count:
                                            self.user_batch_count[user_id] = 0
                                        self.user_batch_count[user_id] += 1
                                        batch_num = self.user_batch_count[user_id]
                                        
                                        # Prepare raw batch data
                                        raw_batch_data = {
                                            "batch_num": batch_num,
                                            "current_frames": len(batch_latencies),
                                            "avg_latency": avg_latency,
                                            "remaining": remaining_frames,
                                            "data_count": len(self.user_latency_history[user_id])
                                        }
                                        
                                        # Store raw data
                                        if user_id not in self.user_raw_data:
                                            self.user_raw_data[user_id] = []
                                        self.user_raw_data[user_id].append(raw_batch_data)
                                        
                                        print(f"[Metrics] Batch {batch_num}/1000: "
                                              f"current_frames={len(batch_latencies)}, "
                                              f"avg_latency={avg_latency:.4f}s, "
                                              f"remaining={remaining_frames}, "
                                              f"data_count={len(self.user_latency_history[user_id])}")
                                        
                                        # Log after 1000 batches
                                        if batch_num >= 1000:
                                            self._log_metrics_to_file(user_id)
                                            # Reset for next 1000 batches
                                            self.user_batch_count[user_id] = 0
                                            self.user_latency_history[user_id] = []
                                            self.user_raw_data[user_id] = []
                        
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

    def _log_metrics_to_file(self, user_id: uuid.UUID):
        """Log metrics to file after collecting 1000 batches"""
        try:
            import json
            import numpy as np
            
            # Get latency history
            if user_id not in self.user_latency_history or len(self.user_latency_history[user_id]) == 0:
                print(f"[Metrics] No latency data to log for user {user_id}")
                return
            
            latencies = np.array(self.user_latency_history[user_id])
            
            # Calculate statistics
            stats = {
                "mean_latency": float(np.mean(latencies)),
                "median_latency": float(np.median(latencies)),
                "p50_latency": float(np.percentile(latencies, 50)),
                "p90_latency": float(np.percentile(latencies, 90)),
                "p95_latency": float(np.percentile(latencies, 95)),
                "p99_latency": float(np.percentile(latencies, 99)),
                "p99_9_latency": float(np.percentile(latencies, 99.9)),
                "min_latency": float(np.min(latencies)),
                "max_latency": float(np.max(latencies)),
                "std_latency": float(np.std(latencies)),
                "sample_count": len(latencies)
            }
            
            # Calculate deadline miss rate using target latency
            deadline = self.target_latency
            missed = np.sum(latencies > deadline)
            deadline_stats = {
                "deadline_seconds": deadline,
                "deadline_miss_rate": float(missed / len(latencies)) if len(latencies) > 0 else 0.0,
                "missed_frames": int(missed),
                "total_frames": len(latencies)
            }
            
            # Calculate jitter (variation in consecutive latencies)
            if len(latencies) > 1:
                jitter = np.abs(np.diff(latencies))
                jitter_stats = {
                    "mean_jitter": float(np.mean(jitter)),
                    "std_jitter": float(np.std(jitter)),
                    "max_jitter": float(np.max(jitter)),
                    "min_jitter": float(np.min(jitter)),
                    "p50_jitter": float(np.percentile(jitter, 50)),
                    "p90_jitter": float(np.percentile(jitter, 90)),
                    "p95_jitter": float(np.percentile(jitter, 95)),
                    "p99_jitter": float(np.percentile(jitter, 99)),
                    "p99.9_jitter": float(np.percentile(jitter, 99.9)),
                    "jitter_variance": float(np.var(jitter))
                }
            else:
                jitter_stats = {}
            
            # Create timestamp folder (YYYYMMDD_HHMM_step{step}_gpu{gpu_ids} format)
            timestamp = time.strftime("%Y%m%d_%H%M")
            # Format GPU IDs: replace commas with underscores for folder naming
            gpu_str = self.gpu_ids.replace(",", "_")
            folder_name = f"{timestamp}_step{self.step}_gpu{gpu_str}"
            session_dir = os.path.join(self.metrics_log_dir, folder_name)
            os.makedirs(session_dir, exist_ok=True)
            
            # Prepare raw data file content
            raw_data_content = {
                "user_id": str(user_id),
                "timestamp": timestamp,
                "target_latency": self.target_latency,
                "batches": self.user_raw_data.get(user_id, [])
            }
            
            # Prepare statistics file content
            statistics_content = {
                "user_id": str(user_id),
                "timestamp": timestamp,
                "target_latency": self.target_latency,
                "batch_count": 1000,
                "total_frames": len(latencies),
                "latency_stats": stats,
                "deadline_miss_rate": deadline_stats,
                "jitter_distribution": jitter_stats,
                "tail_latency": {
                    "p90_latency": stats["p90_latency"],
                    "p95_latency": stats["p95_latency"],
                    "p99_latency": stats["p99_latency"],
                    "p99_9_latency": stats["p99_9_latency"],
                    "max_latency": stats["max_latency"],
                    "mean_latency": stats["mean_latency"],
                    "median_latency": stats["median_latency"]
                }
            }
            
            # Write raw data file
            raw_data_filename = os.path.join(session_dir, f"raw_data_{user_id}.json")
            with open(raw_data_filename, 'w') as f:
                json.dump(raw_data_content, f, indent=2)
            
            # Write statistics file
            statistics_filename = os.path.join(session_dir, f"statistics_{user_id}.json")
            with open(statistics_filename, 'w') as f:
                json.dump(statistics_content, f, indent=2)
            
            print(f"[Metrics] Logged metrics to {session_dir}/")
            print(f"[Metrics]   - Raw data: raw_data_{user_id}.json")
            print(f"[Metrics]   - Statistics: statistics_{user_id}.json")
            print(f"[Metrics] Summary: mean={stats['mean_latency']:.4f}s, "
                  f"p95={stats['p95_latency']:.4f}s, "
                  f"miss_rate={deadline_stats['deadline_miss_rate']*100:.2f}%")
            
        except Exception as e:
            logging.error(f"Error logging metrics to file: {e}")
    
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
