# main.py
# PhysioAI FastAPI backend.
# Runs the RAG pipeline and proxies requests to Ollama.
# Start with: uvicorn main:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import httpx
import json
import os

try:
    # Works when launched as `uvicorn backend.main:app` from project root.
    from .chroma_engine import ChromaRAGEngine
    from .prompts import SYSTEM_PROMPTS
except ImportError:
    # Works when launched as `uvicorn main:app` from inside backend/.
    from chroma_engine import ChromaRAGEngine
    from prompts import SYSTEM_PROMPTS

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="PhysioAI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount frontend ─────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(STATIC_DIR):
    static_assets_dir = os.path.join(STATIC_DIR, "static")
    if os.path.exists(static_assets_dir):
        app.mount("/static", StaticFiles(directory=static_assets_dir), name="static")

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_BACKEND = os.getenv("PHYSIOAI_EMBEDDING_BACKEND", "auto")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.getenv("PHYSIOAI_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
CHROMA_DIR = os.getenv("PHYSIOAI_CHROMA_DIR", os.path.join(PROJECT_ROOT, "chroma_db"))
CHUNK_SIZE = int(os.getenv("PHYSIOAI_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("PHYSIOAI_CHUNK_OVERLAP", "200"))

# ── Build RAG index on startup ─────────────────────────────────────────────────
rag = ChromaRAGEngine(
    data_dir=DATA_DIR,
    persist_dir=CHROMA_DIR,
    ollama_url=OLLAMA_URL,
    embedding_model=EMBEDDING_MODEL,
    embedding_backend=EMBEDDING_BACKEND,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
print(
    f"✅ Chroma RAG ready: {rag.indexed_chunks} chunks from "
    f"{rag.indexed_documents} PDFs"
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    mode: str                         # injury | prevention | nutrition-rec | pre | post
    history: List[Message] = Field(default_factory=list)
    model: Optional[str] = DEFAULT_MODEL
    top_k: Optional[int] = 3

class RetrieveRequest(BaseModel):
    query: str
    mode: str
    top_k: Optional[int] = 3


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend."""
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "PhysioAI API running. Place frontend/index.html to serve the UI."}


@app.get("/health")
async def health():
    """Check if the server and Ollama are reachable."""
    vector_stats = rag.stats()

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        models = []
        ollama_ok = False
    else:
        ollama_ok = True

    return {
        "status": "ok",
        "chunks_indexed": vector_stats["chunks_indexed"],
        "documents_indexed": vector_stats["documents_indexed"],
        "embedding_backend": vector_stats["embedding_backend"],
        "embedding_model": vector_stats["embedding_model"],
        "data_dir": vector_stats["data_dir"],
        "ollama_connected": ollama_ok,
        "ollama_url": OLLAMA_URL,
        "available_models": models,
    }


@app.get("/models")
async def list_models():
    """List models available in Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """
    Run the RAG retrieval step only (no LLM).
    Useful for debugging what chunks are being retrieved.
    """
    if req.mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    results = rag.retrieve(req.query, req.mode, req.top_k or 3)
    return {
        "query": req.query,
        "mode": req.mode,
        "results": [
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "category": chunk["category"],
                "score": round(score * 100, 2),
                "text": chunk["text"],
                "page": chunk.get("page"),
            }
            for chunk, score in results
        ],
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Full RAG + LLM pipeline with streaming response.
    1. Retrieve top_k chunks via Chroma vector search
    2. Inject chunks into system prompt
    3. Stream response from Ollama
    Returns: StreamingResponse with newline-delimited JSON chunks
    """
    if req.mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    # ── Step 1: Retrieve ───────────────────────────────────────────────────────
    results = rag.retrieve(req.query, req.mode, req.top_k or 3)
    context = rag.build_context(results)

    # ── Step 2: Build system prompt with injected context ─────────────────────
    system_prompt = SYSTEM_PROMPTS[req.mode].format(context=context)

    # ── Step 3: Build message list ────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.query})

    # ── Step 4: Stream from Ollama ────────────────────────────────────────────
    retrieved_meta = [
        {
            "id": chunk["id"],
            "source": chunk["source"],
            "category": chunk["category"],
            "score": round(score * 100, 2),
            "text": chunk["text"],
            "page": chunk.get("page"),
        }
        for chunk, score in results
    ]

    async def stream():
        # First yield the retrieved chunks metadata so the frontend
        # can show them in the sidebar immediately
        yield json.dumps({"type": "sources", "sources": retrieved_meta}) + "\n"

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": req.model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 800,
                            "top_p": 0.9,
                        },
                    },
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        yield json.dumps({
                            "type": "error",
                            "message": f"Ollama returned {response.status_code}: {error_body.decode()}"
                        }) + "\n"
                        return

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield json.dumps({"type": "token", "content": token}) + "\n"
                            if data.get("done"):
                                yield json.dumps({"type": "done"}) + "\n"
                                break
                        except json.JSONDecodeError:
                            continue

        except httpx.ConnectError:
            yield json.dumps({
                "type": "error",
                "message": f"Cannot connect to Ollama at {OLLAMA_URL}. Is it running?"
            }) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
