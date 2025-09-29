import asyncio
import multiprocessing as mp
from multiprocessing import Manager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
import io
import os
import sys
from types import SimpleNamespace
import time
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import config


default_prompt = "Realistic images from the sketches, add lifelike details."


def websocket_server(input_queue, output_queue, num_frames_needed):
    app = FastAPI()
    app.state.input_queue = input_queue
    app.state.output_queue = output_queue
    app.state.num_frames_needed = num_frames_needed

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        input_queue = app.state.input_queue
        output_queue = app.state.output_queue
        num_frames_needed = app.state.num_frames_needed
        try:
            while True:
                if input_queue.qsize() >= num_frames_needed.value:
                    await asyncio.sleep(0.01)
                    continue
                await websocket.send_text("input")
                bytes = await websocket.receive_bytes()
                pil_img = Image.open(io.BytesIO(bytes))
                input_queue.put(pil_img)
                if not output_queue.empty():
                    await websocket.send_text("output")
                    out_img = output_queue.get(block=False)
                    buf = io.BytesIO()
                    out_img.save(buf, format="PNG")
                    await websocket.send_bytes(buf.getvalue())
                time.sleep(0.01)

        except WebSocketDisconnect:
            print("Client disconnected")

    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info", reload=False, workers=1)

def generation_loop(config, input_queue, output_queue, num_frames_needed, width = 512, height = 512):
    config.pretty_print()
    if config.use_multi_gpu:
        from vid2vid_pipe import MultiGPUPipeline
        pipeline = MultiGPUPipeline(config)
    else:
        from vid2vid import Pipeline
        pipeline = Pipeline(config)
    first_batch = True
    while True:
        try:
            for _ in range(num_frames_needed.value):
                params = pipeline.InputParams(width=width, height=height, prompt=default_prompt)
                params = SimpleNamespace(**params.model_dump())
                params.image = input_queue.get()
                pipeline.accept_new_params(params)
            for output_image in pipeline.produce_outputs():
                output_queue.put(output_image, block=False)
            if first_batch:
                num_frames_needed.value = 4
                first_batch = False
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    num_frames_needed = mp.Value('i', 5)

    p = mp.Process(target=generation_loop, args=(config, input_queue, output_queue, num_frames_needed))
    p.start()

    # Run FastAPI
    websocket_server(input_queue, output_queue, num_frames_needed)
