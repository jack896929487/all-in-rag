"""
C8 RAG system entrypoint.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    PerformanceMonitor,
    RAGEvaluator,
    RetrievalOptimizationModule,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """Recipe RAG system for C8."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.evaluator: Optional[RAGEvaluator] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.latest_response: Optional[Dict[str, Any]] = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        """Initialize data, index, and generation modules."""
        if all([self.data_module, self.index_module, self.generation_module]):
            return

        print("正在初始化 RAG 系统...")
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )

        print("初始化生成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        print("系统初始化完成。")

    def build_knowledge_base(self):
        """Build or load the knowledge base."""
        if self.retrieval_module is not None:
            return

        if not all([self.data_module, self.index_module, self.generation_module]):
            raise ValueError("请先初始化系统")

        print("\n正在构建知识库...")
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("已加载已有向量索引。")
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("执行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未发现本地索引，开始构建新索引...")
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("执行文本分块...")
            chunks = self.data_module.chunk_documents()
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            print("保存向量索引...")
            self.index_module.save_index()

        print("初始化检索模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        print("\n知识库统计:")
        print(f"  文档总数: {stats['total_documents']}")
        print(f"  文本块数: {stats['total_chunks']}")
        print(f"  菜品分类: {list(stats['categories'].keys())}")
        print(f"  难度分布: {stats['difficulties']}")
        print("知识库构建完成。")

    def setup(self):
        """Initialize all runtime components."""
        if self.performance_monitor is None:
            self.performance_monitor = PerformanceMonitor(
                log_path=self.config.performance_log_path,
            )

        self.initialize_system()
        self.build_knowledge_base()

        if self.evaluator is None:
            self.evaluator = RAGEvaluator(
                self,
                live_eval_log_path=self.config.answer_eval_log_path,
                judge_model_name=self.config.judge_llm_model,
                judge_temperature=self.config.judge_temperature,
                judge_max_tokens=self.config.judge_max_tokens,
                judge_thinking_type=self.config.judge_thinking_type,
            )

    def analyze_and_retrieve(
        self,
        question: str,
        verbose: bool = True,
        performance_trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run query analysis and retrieval without generation."""
        if not all([self.data_module, self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        self._print(verbose, f"\n用户问题: {question}")

        route_stage_start = perf_counter()
        route_type = self.generation_module.query_router(question)
        self._print(verbose, f"查询类型: {route_type}")
        if performance_trace and self.performance_monitor:
            self.performance_monitor.record_stage(
                performance_trace,
                "query_router",
                route_stage_start,
            )

        self._print(verbose, "正在执行查询改写...")
        rewrite_stage_start = perf_counter()
        rewritten_query = self.generation_module.query_rewrite(
            question,
            route_type=route_type,
        )
        self._print(verbose, f"重写后的查询: {rewritten_query}")
        if performance_trace and self.performance_monitor:
            self.performance_monitor.record_stage(
                performance_trace,
                "query_rewrite",
                rewrite_stage_start,
                rewritten_query_length=len(rewritten_query),
            )

        filters = self._extract_filters_from_query(question)
        search_top_k = self._get_search_top_k(route_type)

        retrieval_stage_start = perf_counter()
        if filters:
            self._print(verbose, f"应用元数据过滤: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters,
                top_k=search_top_k,
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=search_top_k,
            )
        if performance_trace and self.performance_monitor:
            self.performance_monitor.record_stage(
                performance_trace,
                "retrieval",
                retrieval_stage_start,
                search_top_k=search_top_k,
                filtered=bool(filters),
            )

        parent_doc_stage_start = perf_counter()
        relevant_docs = (
            self.data_module.get_parent_documents(relevant_chunks)
            if relevant_chunks
            else []
        )
        if performance_trace and self.performance_monitor:
            self.performance_monitor.record_stage(
                performance_trace,
                "parent_doc_assembly",
                parent_doc_stage_start,
            )

        if relevant_chunks:
            self._print(verbose, f"命中文本块数: {len(relevant_chunks)}")
        else:
            self._print(verbose, "未命中相关文本块。")

        retrieved_dishes = self._get_ranked_dish_names(relevant_docs)
        if relevant_docs:
            self._print(verbose, f"命中文档: {', '.join(retrieved_dishes)}")
        elif verbose and relevant_chunks:
            self._print(verbose, "未能还原出完整父文档。")

        if performance_trace and self.performance_monitor:
            self.performance_monitor.set_metadata(
                performance_trace,
                route_type=route_type,
                rewritten_query=rewritten_query,
                filters=filters,
                search_top_k=search_top_k,
                retrieved_chunk_count=len(relevant_chunks),
                retrieved_doc_count=len(relevant_docs),
                retrieved_dishes=retrieved_dishes,
            )

        return {
            "question": question,
            "route_type": route_type,
            "rewritten_query": rewritten_query,
            "filters": filters,
            "relevant_chunks": relevant_chunks,
            "relevant_docs": relevant_docs,
        }

    def generate_answer(
        self,
        question: str,
        route_type: str,
        relevant_docs: List[Any],
        stream: bool = False,
        verbose: bool = True,
    ):
        """Generate final answer from retrieved parent documents."""
        if not self.generation_module:
            raise ValueError("请先初始化生成模块")

        if not relevant_docs:
            return "抱歉，没有找到相关的食谱信息。请尝试换一个问法，或明确菜系、食材、健康目标等条件。"

        if route_type == "list":
            self._print(verbose, "正在使用 LLM 生成推荐回答...")
            if stream:
                return self.generation_module.generate_list_answer_stream(question, relevant_docs)
            return self.generation_module.generate_list_answer(question, relevant_docs)

        self._print(verbose, "正在生成详细回答...")
        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
            return self.generation_module.generate_step_by_step_answer(question, relevant_docs)

        if stream:
            return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
        return self.generation_module.generate_basic_answer(question, relevant_docs)

    def ask_question(self, question: str, stream: bool = False):
        """Full RAG pipeline with automatic answer evaluation and performance monitoring."""
        if not self.performance_monitor or not self.evaluator:
            raise ValueError("请先调用 setup() 初始化系统")

        trace = self.performance_monitor.start_trace(question)
        cache_key = self._normalize_question(question)

        cache_lookup_start = perf_counter()
        cached_response = self.response_cache.get(cache_key)
        self.performance_monitor.record_stage(
            trace,
            "cache_lookup",
            cache_lookup_start,
        )

        if cached_response:
            pipeline = cached_response.get("pipeline", {})
            self.performance_monitor.set_metadata(
                trace,
                cache_hit=True,
                route_type=pipeline.get("route_type"),
                rewritten_query=pipeline.get("rewritten_query"),
                filters=pipeline.get("filters", {}),
                retrieved_chunk_count=len(pipeline.get("relevant_chunks", [])),
                retrieved_doc_count=len(pipeline.get("relevant_docs", [])),
                retrieved_dishes=self._get_ranked_dish_names(pipeline.get("relevant_docs", [])),
                answer_length=len(cached_response.get("answer", "")),
            )
            performance_report = self.performance_monitor.finalize_trace(trace)
            latest_response = dict(cached_response)
            latest_response["performance"] = performance_report
            self.latest_response = latest_response
            self._print(True, "\n命中重复问题缓存，直接返回已有答案。")
            return latest_response["answer"]

        self.performance_monitor.set_metadata(trace, cache_hit=False)
        pipeline = self.analyze_and_retrieve(
            question,
            verbose=True,
            performance_trace=trace,
        )

        generation_stage_start = perf_counter()
        result = self.generate_answer(
            question=question,
            route_type=pipeline["route_type"],
            relevant_docs=pipeline["relevant_docs"],
            stream=stream,
            verbose=True,
        )

        if isinstance(result, str):
            self.performance_monitor.record_stage(
                trace,
                "generation",
                generation_stage_start,
            )
            finalized = self._finalize_response(
                cache_key=cache_key,
                question=question,
                pipeline=pipeline,
                answer=result,
                performance_trace=trace,
            )
            return finalized["answer"]

        return self._stream_and_finalize_response(
            cache_key=cache_key,
            question=question,
            pipeline=pipeline,
            stream_result=result,
            performance_trace=trace,
            generation_stage_start=generation_stage_start,
        )

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """Search dish names by category."""
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        search_query = query or category
        filters = {"category": category}
        docs = self.retrieval_module.metadata_filtered_search(
            search_query,
            filters,
            top_k=10,
        )

        dish_names: List[str] = []
        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        return dish_names

    def get_ingredients_list(self, dish_name: str) -> str:
        """Get ingredient information for a dish."""
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)
        return self.generation_module.generate_basic_answer(
            f"{dish_name}需要什么食材？",
            docs,
        )

    def run_interactive(self):
        """Run interactive Q&A mode."""
        print("=" * 60)
        print("食谱 RAG 系统 - 交互式问答")
        print("=" * 60)

        self.setup()

        print("\n输入问题开始提问，输入 `退出` / `quit` / `exit` 结束。")
        while True:
            try:
                user_input = input("\n你的问题: ").strip()
                if user_input.lower() in ["退出", "quit", "exit", ""]:
                    break

                stream_choice = input("是否使用流式输出? (y/n, 默认 y): ").strip().lower()
                use_stream = stream_choice != "n"

                print("\n回答:")
                result = self.ask_question(user_input, stream=use_stream)
                if use_stream and not isinstance(result, str):
                    for chunk in result:
                        print(chunk, end="", flush=True)
                    print()
                else:
                    print(result)

                self.print_latest_evaluation()
                self.print_latest_performance()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"处理问题时出错: {exc}")

        print("\n已退出食谱 RAG 系统。")

    def print_latest_evaluation(self):
        """Print evaluation for the latest answer."""
        if not self.latest_response or not self.evaluator:
            return

        evaluation = self.latest_response.get("evaluation")
        if evaluation:
            self.evaluator.print_live_evaluation(evaluation)
            retrieval_metrics = evaluation.get("retrieval_metrics", {})
            if retrieval_metrics:
                print("  实时检索指标:")
                print(f"    Recall: {retrieval_metrics.get('recall', 0.0):.4f}")
                print(f"    Precision: {retrieval_metrics.get('precision', 0.0):.4f}")
                print(f"    MRR: {retrieval_metrics.get('mrr', 0.0):.4f}")

    def print_latest_performance(self):
        """Print performance summary for the latest answer."""
        if not self.latest_response or not self.performance_monitor:
            return

        performance = self.latest_response.get("performance")
        if performance:
            self.performance_monitor.print_summary(performance)

    def _extract_filters_from_query(self, query: str) -> Dict[str, str]:
        """Extract metadata filters from the user query."""
        filters: Dict[str, str] = {}

        for category in DataPreparationModule.get_supported_categories():
            if category in query:
                filters["category"] = category
                break

        for difficulty in sorted(
            DataPreparationModule.get_supported_difficulties(),
            key=len,
            reverse=True,
        ):
            if difficulty in query:
                filters["difficulty"] = difficulty
                break

        return filters

    def _get_search_top_k(self, route_type: str) -> int:
        if route_type in {"list", "general"}:
            return max(self.config.top_k, 5)
        return self.config.top_k

    def _finalize_response(
        self,
        cache_key: str,
        question: str,
        pipeline: Dict[str, Any],
        answer: str,
        performance_trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.evaluator or not self.performance_monitor:
            raise ValueError("请先初始化评估模块和性能监控模块")

        evaluation_stage_start = perf_counter()
        evaluation = self.evaluator.evaluate_live_answer(
            question=question,
            answer=answer,
            relevant_docs=pipeline["relevant_docs"],
            route_type=pipeline["route_type"],
            rewritten_query=pipeline["rewritten_query"],
            filters=pipeline["filters"],
            output_path=self.config.answer_eval_log_path,
        )
        self.performance_monitor.record_stage(
            performance_trace,
            "evaluation",
            evaluation_stage_start,
        )

        generation_latency_ms = performance_trace.get("stage_timings_ms", {}).get("generation", 0.0)
        generation_chars_per_second = None
        if generation_latency_ms > 0:
            generation_chars_per_second = round(
                len(answer) / max(generation_latency_ms / 1000.0, 1e-9),
                3,
            )

        self.performance_monitor.set_metadata(
            performance_trace,
            answer_length=len(answer),
            evaluation_overall_score=evaluation["overall_score"],
            evaluation_overall_normalized_score=evaluation["overall_normalized_score"],
            generation_chars_per_second=generation_chars_per_second,
        )
        performance_report = self.performance_monitor.finalize_trace(
            performance_trace,
            output_path=self.config.performance_log_path,
        )

        response_bundle = {
            "question": question,
            "answer": answer,
            "pipeline": pipeline,
            "evaluation": evaluation,
            "performance": performance_report,
        }
        self.response_cache[cache_key] = response_bundle
        self.latest_response = response_bundle
        return response_bundle

    def _stream_and_finalize_response(
        self,
        cache_key: str,
        question: str,
        pipeline: Dict[str, Any],
        stream_result: Iterator[str],
        performance_trace: Dict[str, Any],
        generation_stage_start: float,
    ):
        collected_chunks: List[str] = []
        for chunk in stream_result:
            collected_chunks.append(chunk)
            yield chunk

        self.performance_monitor.record_stage(
            performance_trace,
            "generation",
            generation_stage_start,
        )
        answer = "".join(collected_chunks)
        self._finalize_response(
            cache_key=cache_key,
            question=question,
            pipeline=pipeline,
            answer=answer,
            performance_trace=performance_trace,
        )

    @staticmethod
    def _normalize_question(question: str) -> str:
        return "".join(question.lower().split())

    @staticmethod
    def _get_ranked_dish_names(docs: List[Any]) -> List[str]:
        ranked_names: List[str] = []
        seen = set()
        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            normalized = "".join(dish_name.lower().split())
            if normalized not in seen:
                ranked_names.append(dish_name)
                seen.add(normalized)
        return ranked_names

    @staticmethod
    def _print(enabled: bool, message: str):
        if enabled:
            print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C8 食谱 RAG 系统")
    parser.add_argument(
        "--generate-eval-dataset",
        action="store_true",
        help="基于当前知识库生成默认评估数据集",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="运行 RAG 数据集评估并输出检索/生成质量结果",
    )
    parser.add_argument(
        "--dataset",
        default="./evaluation_dataset.json",
        help="评估数据集路径",
    )
    parser.add_argument(
        "--output",
        default="./evaluation_report.json",
        help="评估报告输出路径",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="自动生成评估集时的样本数",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="评估时仅运行前 N 条样本",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="禁用 LLM-as-Judge，改用启发式评估生成质量",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="即使传入其他参数也强制进入交互模式",
    )
    return parser.parse_args()


def _print_retrieval_metrics_block(evaluation: Dict[str, Any]) -> None:
    """打印检索质量指标。"""
    retrieval_metrics = evaluation.get("retrieval_metrics") or {}
    if not retrieval_metrics:
        return

    recall = retrieval_metrics.get("recall")
    precision = retrieval_metrics.get("precision")
    mrr = retrieval_metrics.get("mrr")
    judge_mode = retrieval_metrics.get("judge_mode")

    print("检索质量指标:")
    if recall is not None:
        print(f"  Recall: {recall:.3f}")
    if precision is not None:
        print(f"  Precision: {precision:.3f}")
    if mrr is not None:
        print(f"  MRR: {mrr:.3f}")
    if judge_mode:
        print(f"  评估方式: {judge_mode}")


def _patched_print_latest_evaluation(self) -> None:
    """打印最近一次答案评估结果，并显示检索指标。"""
    if not self.latest_response:
        return

    evaluation = self.latest_response.get("evaluation")
    if not evaluation:
        return

    judge_model_name = evaluation.get("judge_model_name")
    if judge_model_name:
        print(f"Judge 模型: {judge_model_name}")
    self.evaluator.print_live_evaluation(evaluation)
    _print_retrieval_metrics_block(evaluation)


RecipeRAGSystem.print_latest_evaluation = _patched_print_latest_evaluation


def _ensure_evaluation_reports_dir(self) -> None:
    """确保评估报告目录存在。"""
    import os

    reports_dir = getattr(self.config, "evaluation_reports_dir", "./evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)


def _append_live_evaluation_table(self, response_bundle: Dict[str, Any]) -> None:
    """将单次评估结果追加到汇总表格。"""
    import csv
    import os

    if not response_bundle:
        return

    evaluation = response_bundle.get("evaluation") or {}
    if not evaluation:
        return

    _ensure_evaluation_reports_dir(self)

    retrieval_metrics = evaluation.get("retrieval_metrics") or {}
    generation_metrics = evaluation.get("generation_metrics") or {}
    performance = response_bundle.get("performance") or {}
    retrieved_dishes = (
        response_bundle.get("retrieved_dishes")
        or evaluation.get("retrieved_dishes")
        or []
    )

    factual_accuracy = generation_metrics.get(
        "factual_accuracy",
        evaluation.get("factual_accuracy"),
    )
    faithfulness = generation_metrics.get(
        "faithfulness",
        evaluation.get("faithfulness"),
    )
    relevance = generation_metrics.get(
        "relevance",
        evaluation.get("relevance"),
    )
    overall_score = generation_metrics.get(
        "overall_score",
        evaluation.get("overall_score"),
    )
    generation_judge_mode = generation_metrics.get(
        "judge_mode",
        evaluation.get("judge_mode") or evaluation.get("scoring_method"),
    )

    table_path = getattr(
        self.config,
        "answer_eval_table_path",
        "./evaluation_reports/live_evaluations.csv",
    )
    fieldnames = [
        "timestamp",
        "question",
        "route_type",
        "answer",
        "retrieved_dishes",
        "recall",
        "precision",
        "mrr",
        "retrieval_judge_mode",
        "judge_model_name",
        "factual_accuracy",
        "faithfulness",
        "relevance",
        "overall_score",
        "generation_judge_mode",
        "answer_length",
        "total_latency_seconds",
    ]

    row = {
        "timestamp": evaluation.get("timestamp") or response_bundle.get("timestamp"),
        "question": response_bundle.get("question") or evaluation.get("question"),
        "route_type": response_bundle.get("route_type"),
        "answer": response_bundle.get("answer", ""),
        "retrieved_dishes": "; ".join(retrieved_dishes) if isinstance(retrieved_dishes, list) else str(retrieved_dishes),
        "recall": retrieval_metrics.get("recall"),
        "precision": retrieval_metrics.get("precision"),
        "mrr": retrieval_metrics.get("mrr"),
        "retrieval_judge_mode": retrieval_metrics.get("judge_mode"),
        "judge_model_name": evaluation.get("judge_model_name"),
        "factual_accuracy": factual_accuracy,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "overall_score": overall_score,
        "generation_judge_mode": generation_judge_mode,
        "answer_length": len(response_bundle.get("answer", "")),
        "total_latency_seconds": performance.get("total_duration_seconds"),
    }

    file_exists = os.path.exists(table_path)
    with open(table_path, "a", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


_original_setup = RecipeRAGSystem.setup


def _patched_setup(self) -> bool:
    """初始化系统并创建评估报告目录。"""
    _ensure_evaluation_reports_dir(self)
    return _original_setup(self)


RecipeRAGSystem.setup = _patched_setup


_original_finalize_response = RecipeRAGSystem._finalize_response


def _patched_finalize_response(self, *args, **kwargs):
    """在原有收口逻辑后追加评估表格归档。"""
    response_bundle = _original_finalize_response(self, *args, **kwargs)
    if isinstance(response_bundle, dict):
        _append_live_evaluation_table(self, response_bundle)
    return response_bundle


RecipeRAGSystem._finalize_response = _patched_finalize_response


_original_setup_with_reports = RecipeRAGSystem.setup


def _patched_setup_with_semantic_cache(self) -> bool:
    """初始化系统后启用语义相似缓存。"""
    success = _original_setup_with_reports(self)
    if not success:
        return False

    semantic_cache_enabled = getattr(self.config, "semantic_cache_enabled", True)
    if not semantic_cache_enabled:
        return True

    from rag_modules.semantic_cache import SemanticResponseCache

    model_name = getattr(
        self.config,
        "semantic_cache_embedding_model",
        "BAAI/bge-small-zh-v1.5",
    )
    threshold = getattr(self.config, "semantic_cache_similarity_threshold", 0.88)
    max_entries = getattr(self.config, "semantic_cache_max_entries", 128)

    self.response_cache = SemanticResponseCache(
        model_name=model_name,
        similarity_threshold=threshold,
        max_entries=max_entries,
    )
    return True


RecipeRAGSystem.setup = _patched_setup_with_semantic_cache


_original_ask_question_with_runtime_cache = RecipeRAGSystem.ask_question


def _patched_finalize_response_with_rewritten_cache_key(self, *args, **kwargs):
    """Store responses with the rewritten query as the cache key."""
    pipeline = kwargs.get("pipeline")
    if pipeline is None and len(args) >= 3:
        pipeline = args[2]

    if isinstance(pipeline, dict):
        rewritten_query = pipeline.get("rewritten_query")
        if rewritten_query:
            kwargs["cache_key"] = self._normalize_question(rewritten_query)

    return _patched_finalize_response(self, *args, **kwargs)


def _patched_ask_question_with_rewritten_cache(self, question: str, stream: bool = False):
    """Use rewritten queries as cache keys."""
    if not self.performance_monitor or not self.evaluator:
        raise ValueError("请先调用 setup() 初始化系统")

    trace = self.performance_monitor.start_trace(question)
    pipeline = self.analyze_and_retrieve(
        question,
        verbose=True,
        performance_trace=trace,
    )
    rewritten_query = pipeline.get("rewritten_query") or question
    cache_key = self._normalize_question(rewritten_query)

    cache_lookup_start = perf_counter()
    cached_response = self.response_cache.get(cache_key)
    self.performance_monitor.record_stage(
        trace,
        "cache_lookup",
        cache_lookup_start,
    )

    if cached_response:
        self.performance_monitor.set_metadata(
            trace,
            cache_hit=True,
            route_type=pipeline.get("route_type"),
            rewritten_query=rewritten_query,
            filters=pipeline.get("filters", {}),
            retrieved_chunk_count=len(pipeline.get("relevant_chunks", [])),
            retrieved_doc_count=len(pipeline.get("relevant_docs", [])),
            retrieved_dishes=self._get_ranked_dish_names(pipeline.get("relevant_docs", [])),
            answer_length=len(cached_response.get("answer", "")),
            cache_key=cache_key,
        )
        performance_report = self.performance_monitor.finalize_trace(
            trace,
            output_path=self.config.performance_log_path,
        )
        latest_response = dict(cached_response)
        latest_response["question"] = question
        latest_response["pipeline"] = pipeline
        latest_response["performance"] = performance_report
        self.latest_response = latest_response
        self._print(True, "\n命中改写查询缓存，直接复用已有答案。")
        return latest_response["answer"]

    self.performance_monitor.set_metadata(
        trace,
        cache_hit=False,
        cache_key=cache_key,
    )

    generation_stage_start = perf_counter()
    result = self.generate_answer(
        question=question,
        route_type=pipeline["route_type"],
        relevant_docs=pipeline["relevant_docs"],
        stream=stream,
        verbose=True,
    )

    if isinstance(result, str):
        self.performance_monitor.record_stage(
            trace,
            "generation",
            generation_stage_start,
        )
        finalized = self._finalize_response(
            cache_key=cache_key,
            question=question,
            pipeline=pipeline,
            answer=result,
            performance_trace=trace,
        )
        return finalized["answer"]

    return self._stream_and_finalize_response(
        cache_key=cache_key,
        question=question,
        pipeline=pipeline,
        stream_result=result,
        performance_trace=trace,
        generation_stage_start=generation_stage_start,
    )


RecipeRAGSystem._finalize_response = _patched_finalize_response_with_rewritten_cache_key
RecipeRAGSystem.ask_question = _patched_ask_question_with_rewritten_cache


def main():
    args = parse_args()

    try:
        rag_system = RecipeRAGSystem()

        if args.interactive or not (args.generate_eval_dataset or args.evaluate):
            rag_system.run_interactive()
            return

        rag_system.setup()
        rag_system.evaluator = RAGEvaluator(
            rag_system,
            use_llm_judge=not args.no_llm_judge,
            live_eval_log_path=rag_system.config.answer_eval_log_path,
            judge_model_name=rag_system.config.judge_llm_model,
            judge_temperature=rag_system.config.judge_temperature,
            judge_max_tokens=rag_system.config.judge_max_tokens,
            judge_thinking_type=rag_system.config.judge_thinking_type,
        )

        if args.generate_eval_dataset:
            rag_system.evaluator.generate_default_dataset(
                output_path=args.dataset,
                sample_size=args.sample_size,
            )
            print(f"评估数据集已生成: {args.dataset}")

        if args.evaluate:
            report = rag_system.evaluator.evaluate_dataset(
                dataset_path=args.dataset,
                output_path=args.output,
                sample_limit=args.sample_limit,
            )
            rag_system.evaluator.print_summary(report)
            print(f"\n评估报告已保存: {args.output}")

    except Exception as exc:
        logger.error(f"系统运行出错: {exc}")
        print(f"系统错误: {exc}")


if __name__ == "__main__":
    main()
