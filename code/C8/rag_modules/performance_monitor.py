"""
Performance monitoring utilities for the C8 RAG system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Collect, print, and persist per-request performance traces."""

    STAGE_LABELS = {
        "cache_lookup": "缓存查询",
        "query_router": "查询路由",
        "query_rewrite": "查询改写",
        "retrieval": "检索",
        "parent_doc_assembly": "父文档组装",
        "generation": "答案生成",
        "evaluation": "答案评估",
    }

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path

    def start_trace(self, question: str, event_type: str = "qa_request") -> Dict[str, Any]:
        """Start a new performance trace."""
        return {
            "_start_time": perf_counter(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "question": question,
            "stage_timings_ms": {},
            "stage_metrics": {},
        }

    def record_stage(
        self,
        trace: Dict[str, Any],
        stage_name: str,
        stage_start_time: float,
        **stage_metrics: Any,
    ) -> float:
        """Record one stage timing and optional per-stage metrics."""
        elapsed_ms = round((perf_counter() - stage_start_time) * 1000, 3)
        trace.setdefault("stage_timings_ms", {})[stage_name] = elapsed_ms

        filtered_metrics = {
            key: value
            for key, value in stage_metrics.items()
            if value is not None
        }
        if filtered_metrics:
            trace.setdefault("stage_metrics", {})[stage_name] = filtered_metrics

        return elapsed_ms

    def set_metadata(self, trace: Dict[str, Any], **metadata: Any):
        """Attach top-level metadata to a trace."""
        for key, value in metadata.items():
            if value is not None:
                trace[key] = value

    def finalize_trace(
        self,
        trace: Dict[str, Any],
        output_path: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Finalize a trace and optionally append it to a JSONL log."""
        total_latency_ms = round((perf_counter() - trace["_start_time"]) * 1000, 3)

        report = {
            key: value
            for key, value in trace.items()
            if key != "_start_time"
        }
        report["total_latency_ms"] = total_latency_ms

        target_path = output_path or self.log_path
        if persist and target_path:
            self.append_report(report, target_path)

        return report

    def append_report(self, report: Dict[str, Any], output_path: str):
        """Append one performance report to a JSONL file."""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, "a", encoding="utf-8") as file:
            file.write(json.dumps(report, ensure_ascii=False) + "\n")
        logger.info(f"性能日志已追加到: {output_path_obj}")

    def print_summary(self, report: Dict[str, Any]):
        """Print a compact human-readable performance summary."""
        if not report:
            return

        print("\n性能监控:")
        print(f"  总耗时: {report.get('total_latency_ms', 0.0):.2f} ms")
        print(f"  缓存命中: {'是' if report.get('cache_hit') else '否'}")

        if report.get("route_type"):
            print(f"  路由类型: {report['route_type']}")
        if report.get("retrieved_chunk_count") is not None:
            print(f"  文本块数量: {report['retrieved_chunk_count']}")
        if report.get("retrieved_doc_count") is not None:
            print(f"  文档数量: {report['retrieved_doc_count']}")
        if report.get("answer_length") is not None:
            print(f"  答案长度: {report['answer_length']} 字符")
        if report.get("generation_chars_per_second") is not None:
            print(f"  生成吞吐: {report['generation_chars_per_second']:.2f} 字符/秒")

        stage_timings = report.get("stage_timings_ms", {})
        if stage_timings:
            print("  阶段耗时:")
            for stage_name, elapsed_ms in stage_timings.items():
                label = self.STAGE_LABELS.get(stage_name, stage_name)
                print(f"    {label}: {elapsed_ms:.2f} ms")
