from typing import Dict, List, Union
from uuid import UUID
import asyncio
from fastapi import WebSocket
from starlette.websockets import WebSocketState
import logging
from vid2vid import Pipeline
from types import SimpleNamespace
import time

Connections = Dict[UUID, Dict[str, Union[WebSocket, asyncio.Queue]]]


class ServerFullException(Exception):
    """Exception raised when the server is full."""

    pass


class ConnectionManager:
    def __init__(self):
        self.active_connections: Connections = {}

    async def connect(
        self, user_id: UUID, websocket: WebSocket, max_queue_size: int = 0
    ):
        await websocket.accept()
        user_count = self.get_user_count()
        print(f"[ConnectionManager] User count: {user_count}")
        if max_queue_size > 0 and user_count >= max_queue_size:
            print("[ConnectionManager] Server is full")
            await websocket.send_json({"status": "error", "message": "Server is full"})
            await websocket.close()
            raise ServerFullException("Server is full")
        print(f"[ConnectionManager] New user connected: {user_id}")
        self.active_connections[user_id] = {
            "websocket": websocket,
            "queue": asyncio.Queue(),
            "output_queue": asyncio.Queue(),
            "video_frame_queue": asyncio.Queue(),  # Store all frames from uploaded video
            "video_frames": [],  # Store original video frame data
            "is_upload_mode": False,  # Flag to indicate upload mode
            "video_upload_completed": False,  # Whether the client has finished uploading
            "video_queue_index": 0,  # Current frame index being played
            "video_total_frames": 0,  # Total number of video frames
        }
        await websocket.send_json(
            {"status": "connected", "message": "Connected"},
        )
        await websocket.send_json({"status": "wait"})
        await websocket.send_json({"status": "send_frame"})

    def check_user(self, user_id: UUID) -> bool:
        return user_id in self.active_connections

    async def update_data(self, user_id: UUID, new_data: SimpleNamespace):
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
            await queue.put(new_data)

    async def get_latest_data(self, user_id: UUID) -> SimpleNamespace:
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            try:
                return await queue.get()
            except asyncio.QueueEmpty:
                return None

    def delete_user(self, user_id: UUID):
        print(f"[ConnectionManager] Deleting user: {user_id}")
        user_session = self.active_connections.pop(user_id, None)
        if user_session:
            queue = user_session["queue"]
            output_queue = user_session["output_queue"]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

    def get_user_count(self) -> int:
        return len(self.active_connections)

    def get_websocket(self, user_id: UUID) -> WebSocket:
        user_session = self.active_connections.get(user_id)
        if user_session:
            websocket = user_session["websocket"]
            if websocket.client_state == WebSocketState.CONNECTED:
                return user_session["websocket"]
        return None

    async def disconnect(self, user_id: UUID, pipeline: Pipeline = None):
        print(f"[ConnectionManager] Disconnecting user: {user_id}")
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                await websocket.close()
                print(f"[ConnectionManager] WebSocket closed for user {user_id}")
        except Exception as e:
            logging.error(f"Error: Exception while closing websocket for {user_id}: {e}")
        finally:
            try:
                self.delete_user(user_id)
                print(f"[ConnectionManager] User {user_id} removed from connections")
            except Exception as e:
                logging.error(f"Error: Exception while clearing data for {user_id}: {e}")

    async def disconnect_all(self, pipeline: Pipeline = None):
        """Disconnect all users and close pipeline"""
        print(f"[ConnectionManager] Disconnecting all {len(self.active_connections)} users...")
        user_ids = list(self.active_connections.keys())
        for user_id in user_ids:
            await self.disconnect(user_id, pipeline)
        
        if pipeline:
            try:
                pipeline.close()
                print("[ConnectionManager] Pipeline closed")
            except Exception as e:
                logging.error(f"Error: Exception while closing pipeline: {e}")

    async def send_json(self, user_id: UUID, data: Dict):
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                await websocket.send_json(data)
        except Exception as e:
            logging.error(f"Error: Send json: {e}")
            raise e

    async def receive_json(self, user_id: UUID) -> Dict:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_json()
        except Exception as e:
            logging.error(f"Error: Receive json: {e}")
            raise e

    async def receive_bytes(self, user_id: UUID) -> bytes:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_bytes()
        except Exception as e:
            logging.error(f"Error: Receive bytes: {e}")
            raise e

    async def put_frames_to_output_queue(self, user_id: UUID, frames: List[bytes]):
        session = self.active_connections.get(user_id)
        if session:
            queue = session["output_queue"]
            for frame in frames:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(frame)

    async def get_frame(self, user_id: UUID) -> bytes:
        session = self.active_connections.get(user_id)
        if session:
            queue = session["output_queue"]
            return await queue.get()
        return None

    async def get_output_queue_size(self, user_id: UUID) -> int:
        session = self.active_connections.get(user_id)
        if session:
            queue = session["output_queue"]
            return queue.qsize()
        return 0

    # Set upload mode for user session
    def set_upload_mode(self, user_id: UUID, is_upload: bool):
        session = self.active_connections.get(user_id)
        if session:
            session["is_upload_mode"] = is_upload
            if is_upload:
                # Clear previous video frame data
                session["video_frames"] = []
                session["video_queue_index"] = 0
                session["video_total_frames"] = 0
                session["video_upload_completed"] = False
                # Clear video frame queue
                while not session["video_frame_queue"].empty():
                    try:
                        session["video_frame_queue"].get_nowait()
                    except asyncio.QueueEmpty:
                        break

    # Mark upload completed status
    def set_video_upload_completed(self, user_id: UUID, is_completed: bool):
        session = self.active_connections.get(user_id)
        if session:
            session["video_upload_completed"] = is_completed

    # Check if upload has completed
    def is_video_upload_completed(self, user_id: UUID) -> bool:
        session = self.active_connections.get(user_id)
        if session:
            return session.get("video_upload_completed", False)
        return False

    # Add video frame to queue (upload mode only)
    async def add_video_frame(self, user_id: UUID, frame_data: bytes):
        session = self.active_connections.get(user_id)
        if session and session["is_upload_mode"]:
            # Store original frame data
            session["video_frames"].append(frame_data)
            session["video_total_frames"] = len(session["video_frames"])
            # Add to processing queue
            await session["video_frame_queue"].put(frame_data)

    # Get next video frame with automatic loop (upload mode only)
    async def get_next_video_frame(self, user_id: UUID) -> bytes:
        session = self.active_connections.get(user_id)
        if not session or not session["is_upload_mode"]:
            return None
            
        # If video frame queue is empty, refill with entire video
        if session["video_frame_queue"].empty() and session["video_frames"]:
            print(f"[ConnectionManager] Refilling video queue for user {user_id}, total frames: {session['video_total_frames']}")
            for frame in session["video_frames"]:
                await session["video_frame_queue"].put(frame)
            session["video_queue_index"] = 0
        
        # Get next frame
        if not session["video_frame_queue"].empty():
            frame = await session["video_frame_queue"].get()
            session["video_queue_index"] += 1
            return frame
        
        # If no frames available, wait a bit and try again
        await asyncio.sleep(0)
        return None

    # Get video queue status (upload mode only)
    def get_video_queue_status(self, user_id: UUID) -> dict:
        session = self.active_connections.get(user_id)
        if session and session["is_upload_mode"]:
            return {
                "is_upload_mode": session["is_upload_mode"],
                "video_total_frames": session["video_total_frames"],
                "current_index": session["video_queue_index"],
                "queue_size": session["video_frame_queue"].qsize()
            }
        return {"is_upload_mode": False}
