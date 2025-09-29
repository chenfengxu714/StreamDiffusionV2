import threading
import asyncio
import websockets
import mss
import PIL
from PIL import Image, ImageTk
import io
import tkinter as tk
from multiprocessing import Queue, get_context

top = 0
left = 0

def update_image(label: tk.Label, queue):
    """
    Update the image displayed on a Tkinter label.
    """
    img = queue.get()
    tk_image = ImageTk.PhotoImage(img)
    label.configure(image=tk_image)
    label.image = tk_image  # keep a reference

    # Schedule the next update after 100ms
    label.after(10, update_image, label, queue)


def receive_images(queue: Queue, width: int, height: int) -> None:
    """
    Setup the Tkinter window and start the thread to receive images.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    """
    root = tk.Tk()
    root.title("Image Viewer")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)

    label = tk.Label(root)
    label.pack()

    def on_closing():
        print("window closed")
        root.quit()  # stop event loop
        return

    update_image(label, queue)

    try:
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except KeyboardInterrupt:
        return


def select_monitor_region(width: int, height: int) -> dict:
    global top, left

    root = tk.Tk()
    root.title("Press Enter to start")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)
    root.attributes("-alpha", 0.8)
    root.configure(bg="black")

    def destroy(event=None):
        root.quit()  
        root.after(10, root.destroy)  # destroy safely after mainloop exits

    root.bind("<Return>", destroy)
    root.bind("<Escape>", destroy)

    def update_geometry(event):
        global top, left
        top = root.winfo_y()
        left = root.winfo_x()
    root.bind("<Configure>", update_geometry)

    root.mainloop()
    return {"top": top, "left": left, "width": width, "height": height}


def monitor_setting_process(
    width: int,
    height: int,
    monitor_sender,
) -> None:
    monitor = select_monitor_region(width, height)
    monitor_sender.send(monitor)


async def ws_loop(server_ws: str, stop_event: threading.Event, monitor_receiver: dict, output_queue):
    monitor = monitor_receiver.recv()
    async with websockets.connect(server_ws) as ws:
        with mss.mss() as sct:
            while not stop_event.is_set():
                try:
                    msg = await ws.recv()
                    if msg == "input":
                        # print("received input command")
                        img = sct.grab(monitor)
                        img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        await ws.send(buf.getvalue())
                        # print("sent input")
                    elif msg == "output":
                        # print("received output command")
                        bytes = await ws.recv()
                        pil_img = Image.open(io.BytesIO(bytes))
                        output_queue.put(pil_img)
                        # print("received output")
                except websockets.ConnectionClosed:
                    break


def main(server_ws: str, width=512, height=512):
    ctx = get_context('spawn')
    output_queue = ctx.Queue()
    monitor_sender, monitor_receiver = ctx.Pipe()

    monitor_process = ctx.Process(
        target=monitor_setting_process,
        args=(
            width,
            height,
            monitor_sender,
        ),
    )
    monitor_process.start()
    monitor_process.join()

    viewer_process = ctx.Process(target=receive_images, args=(output_queue, width, height))
    viewer_process.start()

    stop_event = threading.Event()
    try:
        asyncio.run(ws_loop(server_ws, stop_event, monitor_receiver, output_queue))
    except KeyboardInterrupt:
        print("Terminating client...")
    finally:
        stop_event.set()
        viewer_process.terminate()
        viewer_process.join()

if __name__ == "__main__":
    SERVER_WS = "ws://localhost:8888/ws"
    main(SERVER_WS)
