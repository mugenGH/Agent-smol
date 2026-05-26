import json
import numpy as np
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer


MEMORY_PATH = Path(__file__).parent.parent / "memory.json"


@dataclass
class Interaction:
    task: str
    intent: str
    plan_pattern: str
    result_summary: str
    evaluation_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = field(default=None, repr=False)


class MemoryLayer:
    def __init__(self, cfg: dict):
        max_short = cfg.get("max_short_term", 10)
        self.min_score = cfg.get("min_score_to_store", 0.6)
        self.top_k = cfg.get("top_k_retrieval", 3)
        self._short_term: deque[Interaction] = deque(maxlen=max_short)
        self._long_term: List[dict] = self._load()
        try:
            self._embedder = SentenceTransformer(cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        except Exception:
            self._embedder = None

    # ── Short-term ────────────────────────────────────────────────────────────

    def store(self, interaction: Interaction) -> None:
        if self._embedder is not None:
            try:
                text = f"{interaction.task} {interaction.result_summary}"
                interaction.embedding = self._embedder.encode(text).tolist()
            except Exception:
                pass
        self._short_term.append(interaction)
        if interaction.evaluation_score >= self.min_score:
            self._long_term.append(asdict(interaction))
            self._save()

    def get_short_term(self) -> List[Interaction]:
        return list(self._short_term)

    # ── Semantic retrieval ────────────────────────────────────────────────────

    def retrieve_relevant(self, query: str) -> str:
        if not self._long_term:
            return ""

        if self._embedder is None:
            return self._keyword_fallback(query)

        entries_with_embeddings = [e for e in self._long_term if e.get("embedding")]
        if not entries_with_embeddings:
            return self._keyword_fallback(query)

        try:
            query_vec = self._embedder.encode(query)
        except Exception:
            return self._keyword_fallback(query)
        stored_vecs = np.array([e["embedding"] for e in entries_with_embeddings])

        # Cosine similarity
        norms = np.linalg.norm(stored_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        similarities = (stored_vecs / norms) @ query_vec / (np.linalg.norm(query_vec) + 1e-9)

        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        top = [(similarities[i], entries_with_embeddings[i]) for i in top_indices if similarities[i] > 0.3]

        if not top:
            return ""

        lines = ["Relevant past interactions:"]
        for score, entry in top:
            ts = entry["timestamp"][:10]
            lines.append(f"  [{ts}] (sim={score:.2f}) {entry['task']} → {entry['result_summary'][:80]}")
        return "\n".join(lines)

    def recent_context(self, n: int = 3) -> str:
        recent = list(self._short_term)[-n:]
        if not recent:
            return ""
        lines = ["Recent session context:"]
        for i in recent:
            lines.append(f"  - {i.task[:60]} (score: {i.evaluation_score:.1f})")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "avg_score": (
                round(sum(e["evaluation_score"] for e in self._long_term) / len(self._long_term), 2)
                if self._long_term else 0.0
            ),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _keyword_fallback(self, query: str) -> str:
        query_words = set(query.lower().split())
        scored = []
        for entry in self._long_term:
            text = f"{entry['task']} {entry['result_summary']} {' '.join(entry.get('tags', []))}".lower()
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return ""
        lines = ["Relevant past interactions:"]
        for _, entry in scored[:self.top_k]:
            lines.append(f"  - {entry['task']} → {entry['result_summary'][:80]}")
        return "\n".join(lines)

    def _load(self) -> List[dict]:
        if MEMORY_PATH.exists():
            try:
                return json.loads(MEMORY_PATH.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _save(self) -> None:
        MEMORY_PATH.write_text(json.dumps(self._long_term, indent=2))
