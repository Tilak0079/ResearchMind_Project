"""
WebSocket streaming endpoint (Section 7.4): sends status updates and the
final answer back to the client as the pipeline progresses, instead of
one single response at the end like the REST endpoint.

NOTE: true token-by-token streaming (Section 7.4's {"type": "token", ...}
messages) requires the LLM client to support streaming responses -
our current llm_client.py (Phase 12) requests a complete response at
once. This is a real, flagged simplification: we send status updates
as the pipeline progresses, then the FULL answer in one final message,
rather than streaming individual words. Upgrading to true token
streaming is a follow-up, not done silently here.
"""

import logging

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

            await websocket.send_json({"type": "status", "stage": "processing", "detail": "Running guardrails and retrieval..."})

            db = SessionLocal()
            try:
                result = run_query_pipeline(query, db)
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