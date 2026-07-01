from fastapi import FastAPI

from app.catalog import load_catalog
from app.chat import handle_chat
from app.models import ChatRequest, ChatResponse
from app.retriever import HybridRetriever

app = FastAPI(title="SHL Assessment Recommender", version="0.1.0")

CATALOG = load_catalog()
INDEX = HybridRetriever(CATALOG)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "catalog_size": len(CATALOG)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return handle_chat(req.messages, CATALOG, INDEX)
