"""
Configuration for the C8 RAG system.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RAGConfig:
    """Configuration for retrieval, generation, and evaluation."""

    data_path: str = "../../data/C8/cook"
    index_save_path: str = "./vector_index"
    evaluation_reports_dir: str = "./evaluation_reports"
    answer_eval_log_path: str = "./evaluation_reports/answer_evaluation_log.jsonl"
    answer_eval_table_path: str = "./evaluation_reports/live_evaluations.csv"
    performance_log_path: str = "./performance_log.jsonl"
    semantic_cache_enabled: bool = False
    semantic_cache_similarity_threshold: float = 0.88
    semantic_cache_max_entries: int = 128
    semantic_cache_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"
    judge_llm_model: str = "kimi-k2.5"
    judge_temperature: float = 0.6
    judge_max_tokens: int = 4096
    judge_thinking_type: str = "disabled"

    top_k: int = 3

    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "evaluation_reports_dir": self.evaluation_reports_dir,
            "answer_eval_log_path": self.answer_eval_log_path,
            "answer_eval_table_path": self.answer_eval_table_path,
            "performance_log_path": self.performance_log_path,
            "semantic_cache_enabled": self.semantic_cache_enabled,
            "semantic_cache_similarity_threshold": self.semantic_cache_similarity_threshold,
            "semantic_cache_max_entries": self.semantic_cache_max_entries,
            "semantic_cache_embedding_model": self.semantic_cache_embedding_model,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "judge_llm_model": self.judge_llm_model,
            "judge_temperature": self.judge_temperature,
            "judge_max_tokens": self.judge_max_tokens,
            "judge_thinking_type": self.judge_thinking_type,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


DEFAULT_CONFIG = RAGConfig()
