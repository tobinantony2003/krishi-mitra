from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_KB_FILES = [
    "kerala_crops.txt",
    "pest_treatments.txt",
    "fertilizer_guide.txt",
    "govt_schemes.txt",
    "soil_guide.txt",
]


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


def _chunk_text(text: str, chunk_words: int = 200, overlap_words: int = 40) -> List[str]:
    words = re.split(r"\s+", text.strip())
    if not words or words == [""]:
        return []
    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def _naive_retrieve_context(query: str, kb_dir: Path, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Dependency-free fallback when embedding libraries aren't available.

    Scores KB chunks by simple token overlap and returns the top chunks.
    """
    q = (query or "").lower()
    tokens = [t for t in re.split(r"\W+", q) if t]
    scored: List[Tuple[int, str, str]] = []  # (score, chunk, source)

    for fname in DEFAULT_KB_FILES:
        fpath = kb_dir / fname
        if not fpath.exists():
            continue

        text = _read_text(fpath)
        chunks = _chunk_text(text, chunk_words=120, overlap_words=20)
        # Hard cap to keep this fast on low-end machines.
        for chunk in chunks[:200]:
            chunk_l = chunk.lower()
            score = sum(1 for tok in tokens if tok and tok in chunk_l)
            if score > 0:
                scored.append((score, chunk, fname))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, str]] = []
    for score, chunk, source in scored[:top_k]:
        out.append({"text": chunk, "source": source, "score": f"{float(score):.3f}"})

    if out:
        return out

    # If we found no token matches, still return something useful.
    for fname in DEFAULT_KB_FILES:
        fpath = kb_dir / fname
        if not fpath.exists():
            continue
        text = _read_text(fpath)
        chunks = _chunk_text(text, chunk_words=120, overlap_words=20)
        if chunks:
            out.append({"text": chunks[0], "source": fname, "score": "0.000"})
        if len(out) >= top_k:
            break

    return out


@dataclass
class RagConfig:
    knowledge_base_dir: Path
    chroma_dir: Path
    collection_name: str = "krishi_mitra_kb"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_words: int = 200
    overlap_words: int = 40


class RagPipeline:
    def __init__(self, config: RagConfig):
        self.config = config
        self._embedder = None
        self._chroma = None
        self._collection = None

    def _lazy_init(self) -> None:
        if self._collection is not None:
            return

        from sentence_transformers import SentenceTransformer
        import chromadb

        self.config.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = SentenceTransformer(self.config.embedding_model_name)
        self._chroma = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        self._collection = self._chroma.get_or_create_collection(name=self.config.collection_name)

    def build_or_update_index(self) -> Dict[str, int]:
        """
        Ingest all KB text files and store embeddings in ChromaDB.
        Rebuilds ids deterministically from (filename, chunk_index).
        """
        self._lazy_init()
        assert self._collection is not None
        assert self._embedder is not None

        kb_dir = self.config.knowledge_base_dir
        kb_dir.mkdir(parents=True, exist_ok=True)

        stats = {"files": 0, "chunks": 0}
        for fname in DEFAULT_KB_FILES:
            fpath = kb_dir / fname
            if not fpath.exists():
                continue

            text = _read_text(fpath)
            chunks = _chunk_text(text, chunk_words=self.config.chunk_words, overlap_words=self.config.overlap_words)
            if not chunks:
                continue

            ids = [f"{fname}::chunk::{i}" for i in range(len(chunks))]
            metadatas = [{"source": fname, "chunk_index": i} for i in range(len(chunks))]

            # Upsert (delete old ids first to avoid duplicates)
            try:
                self._collection.delete(ids=ids)
            except Exception:
                pass

            embs = self._embedder.encode(chunks, show_progress_bar=False).tolist()
            self._collection.add(ids=ids, documents=chunks, embeddings=embs, metadatas=metadatas)
            stats["files"] += 1
            stats["chunks"] += len(chunks)

        return stats

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        self._lazy_init()
        assert self._collection is not None
        assert self._embedder is not None

        q_emb = self._embedder.encode([query], show_progress_bar=False).tolist()[0]
        res = self._collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[Dict[str, str]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": doc,
                    "source": (meta or {}).get("source", "unknown"),
                    "score": f"{1.0 / (1.0 + float(dist)):.3f}",
                }
            )
        return out

    def _ollama_available(self) -> bool:
        try:
            subprocess.run(
                ["ollama", "--version"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return True
        except Exception:
            return False

    def generate_response(
        self,
        message: str,
        context: List[Dict[str, str]],
        model_name: str = "llama3.2",
        use_ollama_if_available: bool = True,
    ) -> str:
        """
        If Ollama is available locally, calls it. Otherwise returns a grounded,
        rule-based answer using the retrieved context snippets.
        """
        passages = "\n\n".join([f"[{c['source']}] {c['text']}" for c in context])
        prompt = (
            "You are Krishi Mitra, a Kerala farming assistant.\n"
            "Answer with practical steps, short bullet points, and stay grounded in the provided context.\n"
            "If context is insufficient, ask a follow-up question.\n\n"
            f"User message:\n{message}\n\n"
            f"Retrieved context:\n{passages}\n\n"
            "Answer:"
        )

        if use_ollama_if_available and self._ollama_available():
            # Requires: `ollama run llama3.2` etc. on the user's machine (not Colab).
            proc = subprocess.run(
                ["ollama", "run", model_name],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            out = _strip_ansi((proc.stdout or "").strip())
            if out:
                return out

        # Fallback: grounded response
        if not context:
            return (
                "I don’t have enough Kerala-specific context for that yet. "
                "Which crop and district are you asking about, and what is the current season/month?"
            )

        bullets = []
        for c in context[:3]:
            bullets.append(f"- From `{c['source']}`: {c['text'][:260].strip()}...")
        return "Here’s what I found in the knowledge base:\n" + "\n".join(bullets)


def _lazy_translator():
    # Malayalam → English (optional; heavy in Colab). Loaded only when language="ml".
    try:
        from transformers import pipeline
    except ModuleNotFoundError:
        # If translation deps aren't installed, fall back to identity.
        return lambda s: s

    return pipeline("translation", model="Helsinki-NLP/opus-mt-ml-en")


def chat(
    message: str,
    language: str = "en",
    chat_history: Optional[List[Dict[str, str]]] = None,
    knowledge_base_dir: str | Path | None = None,
    chroma_dir: str | Path | None = None,
    top_k: int = 3,
) -> Dict[str, object]:
    kb_dir = Path(knowledge_base_dir) if knowledge_base_dir else (Path(__file__).parent / "knowledge_base")
    ch_dir = Path(chroma_dir) if chroma_dir else (Path(__file__).parent / ".chroma")

    pipeline = RagPipeline(
        RagConfig(
            knowledge_base_dir=kb_dir,
            chroma_dir=ch_dir,
        )
    )

    if language.lower().startswith("ml"):
        translator = _lazy_translator()
        tr_out = translator(message)
        # transformers returns a list of dicts; our identity fallback returns a string.
        if isinstance(tr_out, str):
            message_en = tr_out
        else:
            message_en = tr_out[0]["translation_text"]
    else:
        message_en = message

    try:
        pipeline.build_or_update_index()
        ctx = pipeline.retrieve_context(message_en, top_k=top_k)
    except ModuleNotFoundError:
        # Embedding dependencies not installed; use a dependency-free fallback.
        ctx = _naive_retrieve_context(message_en, kb_dir, top_k=top_k)
    except Exception:
        # Any unexpected retrieval failure should not crash the assistant.
        ctx = _naive_retrieve_context(message_en, kb_dir, top_k=top_k)
    resp = pipeline.generate_response(message_en, ctx)

    sources = sorted({c["source"] for c in ctx})
    followups = [
        "Which district is your farm in?",
        "What is the crop stage and recent rainfall?",
        "Do you prefer organic-only advice or integrated (IPM)?",
    ]

    return {
        "response": resp,
        "sources_used": sources,
        "suggested_followups": followups,
        "language": language,
    }


if __name__ == "__main__":
    # Minimal CLI demo
    out = chat("Best season to plant ginger in Wayanad?", language="en")
    print(json.dumps(out, indent=2, ensure_ascii=False))
