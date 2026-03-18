#!/usr/bin/env bash
# start.sh — Start PhysioAI
# Usage: bash start.sh

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         PhysioAI — Starting          ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Check Python ───────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Install it from python.org"
  exit 1
fi

# ── Check Ollama ───────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "❌  Ollama not found. Install from https://ollama.com"
  exit 1
fi

# ── Start Ollama if not already running ───────────────────────────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "🦙  Starting Ollama..."
  ollama serve &
  sleep 2
else
  echo "🦙  Ollama already running ✓"
fi

# ── Install Python deps if needed ─────────────────────────────────────────────
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "📦  Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "📦  Installing dependencies..."
pip install -q -r requirements.txt

# ── Data and index folders ────────────────────────────────────────────────────
mkdir -p data chroma_db

PDF_COUNT=$(find data -type f -name "*.pdf" | wc -l | tr -d ' ')
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
EMBED_BACKEND="${PHYSIOAI_EMBEDDING_BACKEND:-auto}"

echo "📁  PDF folder: $(pwd)/data"
if [ "${PDF_COUNT}" = "0" ]; then
  echo "ℹ️   No PDFs found yet. Drop files into data/ and restart."
else
  echo "📚  Found ${PDF_COUNT} PDF file(s) in data/"
fi
echo "🧠  Embeddings backend: ${EMBED_BACKEND} (model: ${EMBED_MODEL})"

if ! ollama list | awk 'NR>1 {print $1}' | grep -q "^${EMBED_MODEL}"; then
  echo "ℹ️   Embedding model '${EMBED_MODEL}' is not pulled."
  echo "    Run: ollama pull ${EMBED_MODEL}"
fi

# ── Start FastAPI ──────────────────────────────────────────────────────────────
echo ""
echo "🚀  Starting PhysioAI backend on http://localhost:8000"
echo "🌐  Open your browser at: http://localhost:8000"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

cd backend
uvicorn main:app --reload --port 8000
