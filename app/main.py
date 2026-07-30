"""
FastAPI application entrypoint (Section 7). Run with:
    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI

from app.api.routes_query import router as query_router
from app.api.ws_stream import router as ws_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agentic Hybrid-RAG CS Research Assistant")

app.include_router(query_router)
app.include_router(ws_router)


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is running."""
    return {"status": "ok"}