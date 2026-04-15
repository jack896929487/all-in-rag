"""基于嵌入向量的语义相似缓存。"""

from __future__ import annotations

import logging
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, MutableMapping, Optional

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheEntry:
    """单条缓存记录。"""

    question: str
    embedding: list[float]
    response_bundle: Dict[str, Any]


class SemanticResponseCache(MutableMapping[str, Dict[str, Any]]):
    """支持精确匹配和语义相似匹配的问答缓存。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        similarity_threshold: float = 0.88,
        max_entries: int = 128,
    ) -> None:
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._model: Optional[SentenceTransformer] = None
        self._store: "OrderedDict[str, SemanticCacheEntry]" = OrderedDict()

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("正在初始化语义缓存嵌入模型: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("语义缓存嵌入模型初始化完成")
        return self._model

    def _embed(self, text: str) -> list[float]:
        model = self._load_model()
        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vector.tolist()]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def _clone_response(
        self,
        question: str,
        response_bundle: Dict[str, Any],
        matched_question: str,
        similarity: float,
        match_type: str,
    ) -> Dict[str, Any]:
        result = deepcopy(response_bundle)
        result["question"] = question
        result["cache_hit"] = True
        result["cache_match_type"] = match_type
        result["cache_match_question"] = matched_question
        result["cache_match_similarity"] = round(similarity, 4)
        return result

    def _store_entry(
        self,
        question: str,
        response_bundle: Dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> None:
        if question in self._store:
            self._store.pop(question)

        if embedding is None:
            try:
                embedding = self._embed(question)
            except Exception as exc:
                logger.warning("语义缓存写入时向量化失败，降级为精确缓存: %s", exc)
                embedding = []

        self._store[question] = SemanticCacheEntry(
            question=question,
            embedding=embedding,
            response_bundle=deepcopy(response_bundle),
        )
        self._store.move_to_end(question)

        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def lookup(self, question: str) -> Optional[Dict[str, Any]]:
        if question in self._store:
            entry = self._store.pop(question)
            self._store[question] = entry
            return self._clone_response(
                question=question,
                response_bundle=entry.response_bundle,
                matched_question=question,
                similarity=1.0,
                match_type="exact",
            )

        try:
            query_embedding = self._embed(question)
        except Exception as exc:
            logger.warning("语义缓存查询向量化失败，仅使用精确匹配: %s", exc)
            return None

        best_question: Optional[str] = None
        best_entry: Optional[SemanticCacheEntry] = None
        best_score = -1.0

        for cached_question, entry in self._store.items():
            if not entry.embedding:
                continue
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_question = cached_question
                best_entry = entry

        if (
            best_entry is not None
            and best_question is not None
            and best_score >= self.similarity_threshold
        ):
            logger.info(
                "语义缓存命中: query=%s matched=%s similarity=%.4f",
                question,
                best_question,
                best_score,
            )
            aliased_response = self._clone_response(
                question=question,
                response_bundle=best_entry.response_bundle,
                matched_question=best_question,
                similarity=best_score,
                match_type="semantic",
            )
            self._store_entry(
                question=question,
                response_bundle=best_entry.response_bundle,
                embedding=query_embedding,
            )
            return aliased_response

        return None

    def get(self, key: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        result = self.lookup(key)
        return result if result is not None else default

    def __getitem__(self, key: str) -> Dict[str, Any]:
        result = self.lookup(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self._store_entry(key, value)

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.lookup(key) is not None

    def clear(self) -> None:
        self._store.clear()

