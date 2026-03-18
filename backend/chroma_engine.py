from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
import httpx
from chromadb.config import Settings
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer


ALL_MODES = ("injury", "prevention", "nutrition-rec", "pre", "post")


class ChromaRAGEngine:
    def __init__(
        self,
        data_dir: str,
        persist_dir: str,
        ollama_url: str,
        embedding_model: str = "nomic-embed-text",
        embedding_backend: str = "auto",
        collection_name: str = "physioai_docs",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.ollama_url = ollama_url.rstrip("/")
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = max(300, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 4

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.persist_dir / "index_manifest.json"

        self.hashing_vectorizer = HashingVectorizer(
            n_features=1536,
            norm="l2",
            alternate_sign=False,
        )

        self.embedding_backend = self._select_embedding_backend(embedding_backend)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._get_or_create_collection()

        self.indexed_chunks = 0
        self.indexed_documents = 0
        self._ensure_index()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()

    def _select_embedding_backend(self, requested_backend: str) -> str:
        requested = (requested_backend or "auto").strip().lower()
        if requested not in {"auto", "ollama", "hashing"}:
            requested = "auto"

        if requested == "hashing":
            print("Using local hashing embeddings backend.")
            return "hashing"

        if self._ollama_embeddings_available():
            print(f"Using Ollama embeddings backend with model '{self.embedding_model}'.")
            return "ollama"

        if requested == "ollama":
            raise RuntimeError(
                f"Requested Ollama embeddings, but they are unavailable at {self.ollama_url} "
                f"for model '{self.embedding_model}'."
            )

        print(
            "Ollama embeddings unavailable. Falling back to local hashing embeddings "
            "(lower quality, but fully offline)."
        )
        return "hashing"

    def _ollama_embeddings_available(self) -> bool:
        try:
            _ = self._embed_with_ollama(["embedding-health-check"])
            return True
        except Exception:
            return False

    def _embed_with_ollama(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.embedding_model, "input": texts},
            )

            # Older Ollama versions use /api/embeddings (single prompt).
            if response.status_code == 404:
                embeddings: List[List[float]] = []
                for text in texts:
                    legacy = client.post(
                        f"{self.ollama_url}/api/embeddings",
                        json={"model": self.embedding_model, "prompt": text},
                    )
                    legacy.raise_for_status()
                    payload = legacy.json()
                    vector = payload.get("embedding")
                    if not isinstance(vector, list):
                        raise RuntimeError("Invalid embedding response from Ollama /api/embeddings.")
                    embeddings.append(vector)
                return embeddings

            response.raise_for_status()
            payload = response.json()
            vectors = payload.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise RuntimeError("Invalid embedding response from Ollama /api/embed.")
            return vectors

    def _embed_with_hashing(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        matrix = self.hashing_vectorizer.transform(texts)
        return matrix.astype("float32").toarray().tolist()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_backend == "ollama":
            return self._embed_with_ollama(texts)
        return self._embed_with_hashing(texts)

    def _load_manifest(self) -> Dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_manifest(self, data: Dict):
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _discover_pdf_files(self) -> List[Path]:
        return sorted(self.data_dir.rglob("*.pdf"))

    def _build_signature(self, pdf_files: List[Path]) -> str:
        digest = hashlib.sha256()
        for path in pdf_files:
            stat = path.stat()
            rel = path.relative_to(self.data_dir).as_posix()
            digest.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
        digest.update(f"chunk={self.chunk_size}|overlap={self.chunk_overlap}".encode("utf-8"))
        digest.update(f"embed={self.embedding_backend}|model={self.embedding_model}".encode("utf-8"))
        return digest.hexdigest()

    def _ensure_index(self):
        pdf_files = self._discover_pdf_files()
        self.indexed_documents = len(pdf_files)

        if not pdf_files:
            self.indexed_chunks = self.collection.count()
            if self.indexed_chunks > 0:
                self._reset_collection()
                self.indexed_chunks = 0
            self._write_manifest(
                {
                    "signature": "empty",
                    "pdf_count": 0,
                    "chunk_count": 0,
                    "embedding_backend": self.embedding_backend,
                    "embedding_model": self.embedding_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }
            )
            print(f"No PDFs found in {self.data_dir}. Add files and restart to index.")
            return

        signature = self._build_signature(pdf_files)
        manifest = self._load_manifest()
        current_count = self.collection.count()
        is_stale = (
            manifest.get("signature") != signature
            or manifest.get("chunk_count") != current_count
            or manifest.get("embedding_backend") != self.embedding_backend
            or manifest.get("embedding_model") != self.embedding_model
            or manifest.get("chunk_size") != self.chunk_size
            or manifest.get("chunk_overlap") != self.chunk_overlap
        )

        if is_stale:
            print("PDF corpus changed. Rebuilding Chroma index...")
            self._rebuild_index(pdf_files, signature)
        else:
            self.indexed_chunks = current_count
            print(f"Using existing Chroma index: {self.indexed_chunks} chunks.")

    def _rebuild_index(self, pdf_files: List[Path], signature: str):
        self._reset_collection()

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for pdf_path in pdf_files:
            rel_path = pdf_path.relative_to(self.data_dir).as_posix()
            source_name = pdf_path.name
            mode_tags = "|".join(self._detect_modes(rel_path))

            try:
                reader = PdfReader(str(pdf_path))
            except Exception as exc:
                print(f"Skipping unreadable PDF '{rel_path}': {exc}")
                continue

            for page_number, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                clean_text = self._clean_text(raw_text)
                if len(clean_text) < 40:
                    continue

                chunks = self._split_text(clean_text, self.chunk_size, self.chunk_overlap)
                for chunk_index, chunk_text in enumerate(chunks):
                    chunk_id = self._make_chunk_id(rel_path, page_number, chunk_index, chunk_text)
                    ids.append(chunk_id)
                    documents.append(chunk_text)
                    metadatas.append(
                        {
                            "source": source_name,
                            "source_path": rel_path,
                            "category": f"PDF page {page_number}",
                            "page": page_number,
                            "chunk_index": chunk_index,
                            "mode_tags": mode_tags,
                        }
                    )

        if not documents:
            self.indexed_chunks = 0
            self.indexed_documents = len(pdf_files)
            self._write_manifest(
                {
                    "signature": signature,
                    "pdf_count": len(pdf_files),
                    "chunk_count": 0,
                    "embedding_backend": self.embedding_backend,
                    "embedding_model": self.embedding_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }
            )
            print("No extractable text found in PDFs.")
            return

        batch_size = 32
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            batch_meta = metadatas[start : start + batch_size]
            batch_embeds = self._embed_texts(batch_docs)
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_embeds,
            )

        self.indexed_chunks = self.collection.count()
        self.indexed_documents = len(pdf_files)
        self._write_manifest(
            {
                "signature": signature,
                "pdf_count": len(pdf_files),
                "chunk_count": self.indexed_chunks,
                "embedding_backend": self.embedding_backend,
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        )
        print(
            f"Chroma index built successfully: {self.indexed_chunks} chunks "
            f"from {self.indexed_documents} PDFs."
        )

    def retrieve(self, query: str, mode: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        if self.indexed_chunks == 0:
            return []

        top_k = max(1, int(top_k))
        query_embedding = self._embed_texts([query])[0]
        n_results = min(self.indexed_chunks, max(top_k * 8, top_k))

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        mode_matches: List[Tuple[Dict, float]] = []
        fallback: List[Tuple[Dict, float]] = []

        for doc_id, doc_text, meta, distance in zip(ids, documents, metadatas, distances):
            meta = meta or {}
            score = self._distance_to_score(distance)
            chunk = {
                "id": doc_id,
                "source": meta.get("source", "PDF"),
                "category": meta.get("category", "PDF"),
                "text": doc_text or "",
                "source_path": meta.get("source_path", ""),
                "page": meta.get("page", 0),
            }
            if self._mode_matches(meta.get("mode_tags", ""), mode):
                mode_matches.append((chunk, score))
            else:
                fallback.append((chunk, score))

        if len(mode_matches) < top_k:
            mode_matches.extend(fallback[: top_k - len(mode_matches)])

        return mode_matches[:top_k]

    def build_context(self, results: List[Tuple[Dict, float]]) -> str:
        if not results:
            return "No relevant context found in indexed PDFs."

        blocks = []
        for idx, (chunk, score) in enumerate(results, start=1):
            blocks.append(
                f"[CHUNK {idx}]\n"
                f"Source: {chunk['source']}\n"
                f"Category: {chunk['category']}\n"
                f"Relevance: {score * 100:.1f}%\n\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def stats(self) -> Dict:
        return {
            "chunks_indexed": self.indexed_chunks,
            "documents_indexed": self.indexed_documents,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "data_dir": str(self.data_dir),
            "persist_dir": str(self.persist_dir),
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        current_words: List[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + 1
            if current_words and (current_len + word_len) > chunk_size:
                chunks.append(" ".join(current_words))
                if chunk_overlap > 0:
                    overlap_words: List[str] = []
                    overlap_len = 0
                    for token in reversed(current_words):
                        overlap_words.insert(0, token)
                        overlap_len += len(token) + 1
                        if overlap_len >= chunk_overlap:
                            break
                    current_words = overlap_words
                    current_len = sum(len(token) + 1 for token in current_words)
                else:
                    current_words = []
                    current_len = 0

            current_words.append(word)
            current_len += word_len

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    @staticmethod
    def _make_chunk_id(rel_path: str, page_number: int, chunk_index: int, text: str) -> str:
        raw = f"{rel_path}|{page_number}|{chunk_index}|{text[:80]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _detect_modes(path_str: str) -> List[str]:
        lower = path_str.lower().replace("\\", "/")
        found: List[str] = []
        if "injury" in lower or "rehab" in lower or "recovery" in lower:
            found.append("injury")
        if "prevention" in lower or "prevent" in lower:
            found.append("prevention")
        if "nutrition-rec" in lower or "recovery-nutrition" in lower:
            found.append("nutrition-rec")
        if "pre-workout" in lower or "preworkout" in lower or "/pre/" in lower:
            found.append("pre")
        if "post-workout" in lower or "postworkout" in lower or "/post/" in lower:
            found.append("post")
        return sorted(set(found)) if found else list(ALL_MODES)

    @staticmethod
    def _mode_matches(mode_tags: str, mode: str) -> bool:
        if not mode_tags:
            return True
        tags = {tag.strip() for tag in mode_tags.split("|") if tag.strip()}
        return mode in tags

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        try:
            value = 1.0 - float(distance)
            if value < 0:
                return 0.0
            if value > 1:
                return 1.0
            return value
        except Exception:
            return 0.0
