"""
WebSocket streaming endpoint: sends status updates and the
final answer back to the client as the pipeline progresses, instead of one single response at the end like the REST endpoint.

"""

import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.query_pipeline import run_query_pipeline
from app.db.postgres import SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/v1/query/stream")
async def query_stream(websocket: WebSocket):
    """Handles one WebSocket connection: receives a query, streams status updates back."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            session_id = data.get("session_id") or str(uuid.uuid4())

            await websocket.send_json({"type": "status", "stage": "processing", "detail": "Running guardrails and retrieval..."})

            db = SessionLocal()
            try:
                result = run_query_pipeline(query, db, session_id)
            finally:
                db.close()

            if not result.success:
                await websocket.send_json({"type": "error", "reason": result.error_reason})
                continue

            await websocket.send_json(
                {"type": "route_decision", "route": result.route_taken, "confidence": result.confidence_score}
            )

            await websocket.send_json({"type": "token", "content": result.answer})

            await websocket.send_json({"type": "done", "sources": result.sources})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")