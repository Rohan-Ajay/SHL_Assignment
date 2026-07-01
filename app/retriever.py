import math
import os
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Assessment


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in text.replace("/", " ").replace("-", " ").split() if token.strip()]


@dataclass
class SearchResult:
    assessment: Assessment
    score: float


class HybridRetriever:
    def __init__(self, catalog: list[Assessment]):
        self.catalog = catalog
        self.texts = [item.searchable_text() for item in catalog]
        self._bm25 = BM25Okapi([tokenize(text) for text in self.texts]) if catalog else None
        self._tfidf = TfidfVectorizer(stop_words="english") if catalog else None
        self._tfidf_matrix = self._tfidf.fit_transform(self.texts) if catalog else None
        self._embedder = None
        self._embeddings = None
        self._faiss_index = None
        self._load_embedder()

    def _load_embedder(self) -> None:
        if not self.catalog or os.getenv("USE_SENTENCE_TRANSFORMERS", "").casefold() not in {"1", "true", "yes"}:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._embeddings = self._embedder.encode(self.texts, normalize_embeddings=True)
            try:
                import faiss

                dimension = int(self._embeddings.shape[1])
                self._faiss_index = faiss.IndexFlatIP(dimension)
                self._faiss_index.add(np.asarray(self._embeddings, dtype="float32"))
            except Exception:
                self._faiss_index = None
        except Exception:
            self._embedder = None
            self._embeddings = None
            self._faiss_index = None

    def search(self, query: str, k: int = 15) -> list[SearchResult]:
        if not self.catalog:
            return []

        bm25_scores = self._bm25.get_scores(tokenize(query)) if self._bm25 else np.zeros(len(self.catalog))
        bm25_norm = self._normalize(bm25_scores)

        if self._embedder and self._embeddings is not None:
            query_vec = self._embedder.encode([query], normalize_embeddings=True)[0]
            if self._faiss_index:
                semantic_scores = np.zeros(len(self.catalog))
                scores, indexes = self._faiss_index.search(np.asarray([query_vec], dtype="float32"), len(self.catalog))
                for score, index in zip(scores[0], indexes[0]):
                    if index >= 0:
                        semantic_scores[index] = score
            else:
                semantic_scores = np.dot(self._embeddings, query_vec)
        elif self._tfidf and self._tfidf_matrix is not None:
            query_vec = self._tfidf.transform([query])
            semantic_scores = cosine_similarity(self._tfidf_matrix, query_vec).ravel()
        else:
            semantic_scores = np.zeros(len(self.catalog))
        semantic_norm = self._normalize(semantic_scores)

        combined = (0.55 * semantic_norm) + (0.45 * bm25_norm)
        combined += self._domain_boosts(query)
        if combined.size == 0 or combined.max() <= 0:
            return []
        ranked = sorted(enumerate(combined), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(assessment=self.catalog[index], score=float(score))
            for index, score in ranked[:k]
            if score > 0
        ]

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return scores
        minimum = scores.min()
        maximum = scores.max()
        if math.isclose(maximum, minimum):
            return np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)

    def _domain_boosts(self, query: str) -> np.ndarray:
        query_folded = query.casefold()
        boosts = np.zeros(len(self.catalog))
        technical_query = any(
            token in query_folded
            for token in [
                "developer",
                "engineer",
                "backend",
                "front-end",
                "frontend",
                "java",
                "python",
                "javascript",
                "coding",
                "programming",
                "software",
            ]
        )
        if technical_query:
            for index, item in enumerate(self.catalog):
                text = item.searchable_text().casefold()
                if any(token in text for token in ["coding", "technical", "programming", "backend", "front-end", "database"]):
                    boosts[index] += 0.35
                if any(token in text for token in ["language proficiency", "spoken language", "grammar", "listening"]):
                    boosts[index] -= 0.2
        return boosts
