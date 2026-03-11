from fastapi import FastAPI

from app.api.controllers.chat_controller import router as chat_router
from app.api.controllers.health_controller import router as health_router

app = FastAPI(
    title="Transcript RAG API",
    version="1.0.0",
    description="RAG API for Acquired podcast transcripts",
)

app.include_router(health_router)
app.include_router(chat_router)