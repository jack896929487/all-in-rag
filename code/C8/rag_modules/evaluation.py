"""
Evaluation utilities for the C8 RAG system.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """One evaluation sample."""

    question: str
    expected_relevant_dishes: List[str]
    reference_answer: str = ""
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSample":
        return cls(
            question=data["question"],
            expected_relevant_dishes=data.get("expected_relevant_dishes", []),
            reference_answer=data.get("reference_answer", ""),
            metadata=data.get("metadata", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "expected_relevant_dishes": self.expected_relevant_dishes,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata or {},
        }


class RAGEvaluator:
    """Evaluator for retrieval quality and generation quality."""

    GENERATION_METRICS = ("factual_accuracy", "faithfulness", "relevance")

    RUBRIC = """
你需要从三个维度对 RAG 系统回答进行评分，每个维度输出 1-5 分：

1. factual_accuracy
- 5分：回答中的事实与检索证据、参考答案完全一致，没有明显错误
- 4分：大部分事实正确，只有轻微遗漏或不影响结论的小偏差
- 3分：整体方向基本正确，但存在可见遗漏、模糊表达或局部不准确
- 2分：包含多个事实性问题，回答可靠性较弱
- 1分：回答大部分不正确，或与问题、证据明显冲突

2. faithfulness
- 5分：回答严格基于检索证据，没有编造
- 4分：基本基于检索证据，只有极少量合理扩展
- 3分：部分内容可由证据支持，但存在不确定扩展
- 2分：较多内容无法从证据中找到支持
- 1分：大量编造，和检索证据脱节

3. relevance
- 5分：回答完整回应问题，重点准确
- 4分：回答基本相关，只有少量冗余或轻微遗漏
- 3分：回答部分相关，但重点不够集中
- 2分：回答与问题只有弱相关
- 1分：回答基本答非所问
""".strip()

    def __init__(
        self,
        rag_system,
        use_llm_judge: bool = True,
        live_eval_log_path: Optional[str] = None,
        judge_model_name: Optional[str] = None,
        judge_temperature: Optional[float] = None,
        judge_max_tokens: Optional[int] = None,
        judge_thinking_type: str = "disabled",
    ):
        self.rag_system = rag_system
        self.use_llm_judge = use_llm_judge
        self.live_eval_log_path = live_eval_log_path
        self.judge_model_name = (
            judge_model_name
            or getattr(getattr(self.rag_system, "config", None), "judge_llm_model", None)
            or getattr(getattr(self.rag_system, "config", None), "llm_model", None)
        )
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens
        self.judge_thinking_type = judge_thinking_type
        self.judge_llm = self._build_judge_llm()
        self._candidate_dish_catalog: Optional[List[Dict[str, str]]] = None

    def _build_judge_llm(self):
        """Build a dedicated LLM instance for LLM-as-Judge."""
        if not self.use_llm_judge:
            return None

        api_key = os.getenv("MOONSHOT_JUDGE_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        if not api_key or not self.judge_model_name:
            logger.warning("未找到可用的 Judge 模型配置或 API Key，将回退到启发式评估。")
            return None

        model_kwargs: Dict[str, Any] = {}
        resolved_temperature = self.judge_temperature
        resolved_max_tokens = self.judge_max_tokens

        if self.judge_model_name.startswith("kimi-k2.5"):
            resolved_temperature = (
                resolved_temperature
                if resolved_temperature is not None
                else (0.6 if self.judge_thinking_type == "disabled" else 1.0)
            )
            resolved_max_tokens = resolved_max_tokens or 4096
            model_kwargs["thinking"] = {"type": self.judge_thinking_type}
        else:
            resolved_temperature = resolved_temperature if resolved_temperature is not None else 0.1
            resolved_max_tokens = resolved_max_tokens or 2048

        logger.info(
            "正在初始化 LLM-as-Judge 模型: %s (temperature=%s, thinking=%s)",
            self.judge_model_name,
            resolved_temperature,
            self.judge_thinking_type,
        )
        return MoonshotChat(
            model=self.judge_model_name,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            model_kwargs=model_kwargs,
            moonshot_api_key=api_key,
        )

    def evaluate_live_answer(
        self,
        question: str,
        answer: str,
        relevant_docs: List[Document],
        route_type: str = "general",
        rewritten_query: str = "",
        filters: Optional[Dict[str, Any]] = None,
        reference_answer: str = "",
        expected_relevant_dishes: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate one generated answer in a live conversation."""
        retrieved_dishes = self._get_ranked_dish_names(relevant_docs)
        sample = EvaluationSample(
            question=question,
            expected_relevant_dishes=expected_relevant_dishes or [],
            reference_answer=reference_answer,
            metadata={"route_type": route_type, "filters": filters or {}},
        )
        retrieval_metrics = self._evaluate_live_retrieval(
            question=question,
            relevant_docs=relevant_docs,
            expected_relevant_dishes=expected_relevant_dishes or [],
        )
        generation_metrics = self._evaluate_generation(
            sample=sample,
            answer=answer,
            relevant_docs=relevant_docs,
        )

        scores = [
            generation_metrics[metric_name]["score"]
            for metric_name in self.GENERATION_METRICS
        ]
        normalized_scores = [
            generation_metrics[metric_name]["normalized_score"]
            for metric_name in self.GENERATION_METRICS
        ]

        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "route_type": route_type,
            "rewritten_query": rewritten_query,
            "judge_model_name": self.judge_model_name,
            "filters": filters or {},
            "retrieved_dishes": retrieved_dishes,
            "retrieved_doc_count": len(relevant_docs),
            "retrieval_metrics": retrieval_metrics,
            "answer": answer,
            "generation_metrics": generation_metrics,
            "overall_score": round(mean(scores), 4) if scores else 0.0,
            "overall_normalized_score": round(mean(normalized_scores), 4) if normalized_scores else 0.0,
        }

        target_path = output_path or self.live_eval_log_path
        if persist and target_path:
            self.append_live_evaluation(report, target_path)

        return report

    def append_live_evaluation(self, report: Dict[str, Any], output_path: str):
        """Append one live evaluation result to a JSONL file."""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, "a", encoding="utf-8") as file:
            file.write(json.dumps(report, ensure_ascii=False) + "\n")
        logger.info(f"单次回答评估已追加到: {output_path_obj}")

    def print_live_evaluation(self, report: Dict[str, Any]):
        """Print compact evaluation for one answer."""
        if not report:
            return

        generation_metrics = report["generation_metrics"]
        judge_mode = generation_metrics.get("judge_mode", "unknown")
        retrieved_dishes = report.get("retrieved_dishes", [])

        print("\n回答评估:")
        print(f"  评估方式: {judge_mode}")
        print(
            f"  综合得分: {report['overall_score']:.2f}/5 "
            f"(归一化 {report['overall_normalized_score']:.4f})"
        )
        if retrieved_dishes:
            print(f"  参考菜品: {', '.join(retrieved_dishes)}")
        else:
            print("  参考菜品: 无")

        label_mapping = {
            "factual_accuracy": "事实准确性",
            "faithfulness": "忠实度",
            "relevance": "相关性",
        }
        for metric_name in self.GENERATION_METRICS:
            payload = generation_metrics[metric_name]
            label = label_mapping[metric_name]
            print(f"  {label}: {payload['score']}/5")
            print(f"    说明: {payload['reason']}")

    def generate_default_dataset(self, output_path: str, sample_size: int = 12) -> List[Dict[str, Any]]:
        """Generate a starter evaluation dataset from current documents."""
        if not self.rag_system.data_module or not self.rag_system.data_module.documents:
            raise ValueError("请先初始化系统并构建知识库，再生成评估数据集。")

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        documents = self.rag_system.data_module.documents
        grouped_docs: Dict[str, List[Document]] = {}
        for doc in documents:
            category = doc.metadata.get("category", "其他")
            grouped_docs.setdefault(category, []).append(doc)

        ordered_categories = sorted(grouped_docs.keys())
        selected_docs: List[Document] = []
        round_index = 0
        target_size = min(sample_size, len(documents))

        while len(selected_docs) < target_size:
            progressed = False
            for category in ordered_categories:
                docs_in_category = grouped_docs[category]
                if round_index < len(docs_in_category):
                    selected_docs.append(docs_in_category[round_index])
                    progressed = True
                    if len(selected_docs) >= target_size:
                        break
            if not progressed:
                break
            round_index += 1

        question_templates = [
            ("detail", "{dish_name}怎么做？"),
            ("ingredients", "{dish_name}需要什么食材？"),
            ("detail", "{dish_name}的制作步骤是什么？"),
        ]

        samples: List[Dict[str, Any]] = []
        for index, doc in enumerate(selected_docs):
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            category = doc.metadata.get("category", "其他")
            difficulty = doc.metadata.get("difficulty", "未知")
            question_type, template = question_templates[index % len(question_templates)]

            samples.append(
                {
                    "question": template.format(dish_name=dish_name),
                    "expected_relevant_dishes": [dish_name],
                    "reference_answer": self._build_reference_answer(doc),
                    "metadata": {
                        "dish_name": dish_name,
                        "category": category,
                        "difficulty": difficulty,
                        "question_type": question_type,
                    },
                }
            )

        dataset = {
            "version": "1.0",
            "description": "C8 自动生成评估数据集，可继续补充 reference_answer 和 expected_relevant_dishes。",
            "samples": samples,
        }

        with open(output_path_obj, "w", encoding="utf-8") as file:
            json.dump(dataset, file, ensure_ascii=False, indent=2)

        logger.info(f"评估数据集已生成: {output_path_obj}")
        return samples

    def evaluate_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run end-to-end evaluation on a dataset."""
        samples = self._load_dataset(dataset_path)
        if sample_limit is not None:
            samples = samples[:sample_limit]

        logger.info(f"开始评估数据集，共 {len(samples)} 条样本")
        sample_reports: List[Dict[str, Any]] = []

        for index, sample in enumerate(samples, 1):
            logger.info(f"评估样本 {index}/{len(samples)}: {sample.question}")
            pipeline = self.rag_system.analyze_and_retrieve(sample.question, verbose=False)
            relevant_docs = pipeline["relevant_docs"]
            retrieved_dishes = self._get_ranked_dish_names(relevant_docs)

            try:
                answer = ""
                if relevant_docs:
                    answer = self.rag_system.generate_answer(
                        question=sample.question,
                        route_type=pipeline["route_type"],
                        relevant_docs=relevant_docs,
                        stream=False,
                        verbose=False,
                    )
            except Exception as exc:
                logger.warning(f"样本回答生成失败，记为空字符串: {exc}")
                answer = ""

            retrieval_metrics = self._evaluate_retrieval(
                expected_dishes=sample.expected_relevant_dishes,
                retrieved_dishes=retrieved_dishes,
            )
            live_report = self.evaluate_live_answer(
                question=sample.question,
                answer=answer,
                relevant_docs=relevant_docs,
                route_type=pipeline["route_type"],
                rewritten_query=pipeline["rewritten_query"],
                filters=pipeline["filters"],
                reference_answer=sample.reference_answer,
                expected_relevant_dishes=sample.expected_relevant_dishes,
                output_path=None,
                persist=False,
            )

            sample_reports.append(
                {
                    "question": sample.question,
                    "metadata": sample.metadata or {},
                    "route_type": pipeline["route_type"],
                    "rewritten_query": pipeline["rewritten_query"],
                    "expected_relevant_dishes": sample.expected_relevant_dishes,
                    "retrieved_dishes": retrieved_dishes,
                    "retrieval_metrics": retrieval_metrics,
                    "answer": answer,
                    "generation_metrics": live_report["generation_metrics"],
                    "generation_overall_score": live_report["overall_score"],
                    "generation_overall_normalized_score": live_report["overall_normalized_score"],
                }
            )

        report = {
            "dataset_path": str(Path(dataset_path).resolve()),
            "num_samples": len(sample_reports),
            "retrieval_summary": self._summarize_retrieval(sample_reports),
            "generation_summary": self._summarize_generation(sample_reports),
            "samples": sample_reports,
        }

        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path_obj, "w", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
            logger.info(f"评估报告已保存: {output_path_obj}")

        return report

    def print_summary(self, report: Dict[str, Any]):
        """Print compact dataset evaluation summary."""
        retrieval = report["retrieval_summary"]
        generation = report["generation_summary"]

        print("\n" + "=" * 60)
        print("RAG 评估结果")
        print("=" * 60)
        print(f"样本数: {report['num_samples']}")
        print("\n检索质量:")
        print(f"  Recall 平均值: {retrieval['avg_recall']:.4f}")
        print(f"  Precision 平均值: {retrieval['avg_precision']:.4f}")
        print(f"  MRR 平均值: {retrieval['avg_mrr']:.4f}")

        print("\n生成质量:")
        label_mapping = {
            "avg_factual_accuracy": "事实准确性",
            "avg_faithfulness": "忠实度",
            "avg_relevance": "相关性",
            "avg_overall_generation": "综合得分",
        }
        for metric_name, metric_value in generation.items():
            label = label_mapping.get(metric_name, metric_name)
            print(
                f"  {label}: {metric_value['avg_score']:.2f}/5 "
                f"(归一化 {metric_value['avg_normalized_score']:.4f})"
            )

    def _load_dataset(self, dataset_path: str) -> List[EvaluationSample]:
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            raise FileNotFoundError(f"评估数据集不存在: {dataset_path}")

        with open(dataset_path_obj, "r", encoding="utf-8") as file:
            raw = json.load(file)

        if isinstance(raw, list):
            return [EvaluationSample.from_dict(item) for item in raw]
        return [EvaluationSample.from_dict(item) for item in raw.get("samples", [])]

    def _evaluate_live_retrieval(
        self,
        question: str,
        relevant_docs: List[Document],
        expected_relevant_dishes: List[str],
    ) -> Dict[str, Any]:
        retrieved_dishes = self._get_ranked_dish_names(relevant_docs)

        if expected_relevant_dishes:
            metrics = self._evaluate_retrieval(expected_relevant_dishes, retrieved_dishes)
            metrics["judge_mode"] = "labeled_dataset"
            metrics["expected_relevant_dishes"] = expected_relevant_dishes
            metrics["reason"] = "基于标注数据直接计算检索指标。"
            return metrics

        judged_result = self._infer_relevant_dishes_for_live_question(question)
        metrics = self._evaluate_retrieval(
            judged_result["relevant_dishes"],
            retrieved_dishes,
        )
        metrics["judge_mode"] = judged_result["judge_mode"]
        metrics["expected_relevant_dishes"] = judged_result["relevant_dishes"]
        metrics["reason"] = judged_result.get("reason", "")
        return metrics

    def _infer_relevant_dishes_for_live_question(self, question: str) -> Dict[str, Any]:
        if self.use_llm_judge and self.judge_llm:
            try:
                return self._infer_relevant_dishes_with_llm(question)
            except Exception as exc:
                logger.warning(f"实时检索评估的 LLM 判定失败，回退到启发式规则: {exc}")

        return self._infer_relevant_dishes_with_heuristics(question)

    def _infer_relevant_dishes_with_llm(self, question: str) -> Dict[str, Any]:
        catalog = self._get_candidate_dish_catalog()
        if not catalog:
            return {
                "relevant_dishes": [],
                "judge_mode": "empty_catalog",
                "reason": "知识库中没有可用的菜品目录。",
            }

        catalog_text = "\n".join(
            f"{index + 1}. {item['dish_name']} | {item['category']}"
            for index, item in enumerate(catalog)
        )
        prompt = ChatPromptTemplate.from_template(
            """
你是一个 RAG 检索评估器。请根据用户问题，从候选菜品列表中选出“理论上应该被检索到、能够帮助回答问题”的菜品名称。

用户问题:
{question}

候选菜品列表:
{catalog}

要求:
1. 只能选择候选列表中已有的菜品名称。
2. 推荐类问题可以返回多个相关菜品，但请保持保守，最多返回 8 个。
3. 如果候选列表中没有明显合适的菜品，可以返回空列表。
4. 只输出 JSON，不要额外解释，不要 markdown 代码块。

输出格式:
{{
  "relevant_dishes": ["菜品1", "菜品2"],
  "reason": "简要说明判定依据"
}}
""".strip()
        )

        chain = prompt | self.judge_llm | StrOutputParser()
        raw_response = chain.invoke(
            {
                "question": question,
                "catalog": catalog_text,
            }
        )
        parsed = self._parse_json_response(raw_response)

        candidate_names = {
            self._normalize_name(item["dish_name"]): item["dish_name"]
            for item in catalog
        }
        relevant_dishes: List[str] = []
        for dish_name in parsed.get("relevant_dishes", []):
            normalized_name = self._normalize_name(str(dish_name))
            if normalized_name in candidate_names and candidate_names[normalized_name] not in relevant_dishes:
                relevant_dishes.append(candidate_names[normalized_name])

        return {
            "relevant_dishes": relevant_dishes,
            "judge_mode": "llm_as_judge",
            "reason": str(parsed.get("reason", "")).strip(),
        }

    def _infer_relevant_dishes_with_heuristics(self, question: str) -> Dict[str, Any]:
        catalog = self._get_candidate_dish_catalog()
        question_terms = set(self._extract_terms(question))

        scored_items = []
        for item in catalog:
            candidate_text = f"{item['dish_name']} {item['category']}"
            candidate_terms = set(self._extract_terms(candidate_text))
            overlap = len(question_terms.intersection(candidate_terms))
            if overlap > 0:
                scored_items.append((overlap, item["dish_name"]))

        scored_items.sort(key=lambda item: item[0], reverse=True)
        relevant_dishes = []
        for _, dish_name in scored_items[:8]:
            if dish_name not in relevant_dishes:
                relevant_dishes.append(dish_name)

        return {
            "relevant_dishes": relevant_dishes,
            "judge_mode": "heuristic_fallback",
            "reason": "基于问题词项与菜品名称/分类的重合度估计理论相关菜品。",
        }

    def _get_candidate_dish_catalog(self) -> List[Dict[str, str]]:
        if self._candidate_dish_catalog is not None:
            return self._candidate_dish_catalog

        catalog: List[Dict[str, str]] = []
        seen = set()
        documents = getattr(self.rag_system.data_module, "documents", []) or []
        for doc in documents:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            category = doc.metadata.get("category", "未知分类")
            normalized_name = self._normalize_name(dish_name)
            if normalized_name in seen:
                continue
            seen.add(normalized_name)
            catalog.append(
                {
                    "dish_name": dish_name,
                    "category": category,
                }
            )

        self._candidate_dish_catalog = catalog
        return catalog

    def _evaluate_retrieval(self, expected_dishes: List[str], retrieved_dishes: List[str]) -> Dict[str, Any]:
        expected_norm = [self._normalize_name(name) for name in expected_dishes]
        retrieved_norm = [self._normalize_name(name) for name in retrieved_dishes]

        expected_set = set(expected_norm)
        retrieved_set = set(retrieved_norm)
        hits = expected_set.intersection(retrieved_set)

        recall = len(hits) / len(expected_set) if expected_set else 0.0
        precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
        mrr = 0.0

        for rank, dish in enumerate(retrieved_norm, 1):
            if dish in expected_set:
                mrr = 1.0 / rank
                break

        return {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "mrr": round(mrr, 4),
            "hits": len(hits),
            "num_expected": len(expected_set),
            "num_retrieved": len(retrieved_set),
        }

    def _evaluate_generation(
        self,
        sample: EvaluationSample,
        answer: str,
        relevant_docs: List[Document],
    ) -> Dict[str, Any]:
        context = self._build_context_for_judge(relevant_docs)

        if self.use_llm_judge and self.judge_llm:
            try:
                result = self._judge_with_llm(
                    question=sample.question,
                    answer=answer,
                    reference_answer=sample.reference_answer,
                    context=context,
                )
                result["judge_mode"] = "llm_as_judge"
                return result
            except Exception as exc:
                logger.warning(f"LLM-as-Judge 失败，回退到启发式评估: {exc}")

        result = self._judge_with_heuristics(
            question=sample.question,
            answer=answer,
            reference_answer=sample.reference_answer,
            context=context,
        )
        result["judge_mode"] = "heuristic_fallback"
        return result

    def _judge_with_llm(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        context: str,
    ) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template(
            """
你是一个严格的 RAG 评估器。请根据给定评分标准，对模型回答在以下三个维度打分：
1. factual_accuracy
2. faithfulness
3. relevance

评分标准：
{rubric}

输入信息：
- 用户问题：{question}
- 参考答案：{reference_answer}
- 检索证据：{context}
- 模型回答：{answer}

要求：
1. 每个维度只打 1-5 分的整数。
2. reason 要简洁、可解释。
3. 只输出 JSON，不要额外说明，不要 markdown 代码块。
4. 如果没有参考答案，请优先依据检索证据判断 factual_accuracy。

输出格式：
{{
  "factual_accuracy": {{"score": 1, "reason": "..." }},
  "faithfulness": {{"score": 1, "reason": "..." }},
  "relevance": {{"score": 1, "reason": "..." }}
}}
""".strip()
        )

        chain = prompt | self.judge_llm | StrOutputParser()
        raw_response = chain.invoke(
            {
                "rubric": self.RUBRIC,
                "question": question,
                "reference_answer": reference_answer or "未提供参考答案，请优先依据检索证据评分。",
                "context": context or "未检索到有效证据。",
                "answer": answer or "空回答",
            }
        )
        return self._format_generation_scores(self._parse_json_response(raw_response))

    def _judge_with_heuristics(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        context: str,
    ) -> Dict[str, Any]:
        answer_reference_overlap = self._text_similarity(answer, reference_answer) if reference_answer else 0.0
        answer_context_overlap = self._text_similarity(answer, context)
        question_answer_overlap = self._text_similarity(question, answer)

        factual_basis = (
            0.65 * answer_reference_overlap + 0.35 * answer_context_overlap
            if reference_answer
            else answer_context_overlap
        )
        faithfulness_basis = answer_context_overlap
        relevance_basis = (
            0.7 * question_answer_overlap + 0.3 * answer_reference_overlap
            if reference_answer
            else question_answer_overlap
        )

        return {
            "factual_accuracy": self._basis_metric(
                factual_basis,
                f"启发式评分：回答与参考答案/证据的重合度为 {factual_basis:.4f}",
            ),
            "faithfulness": self._basis_metric(
                faithfulness_basis,
                f"启发式评分：回答与检索证据的重合度为 {faithfulness_basis:.4f}",
            ),
            "relevance": self._basis_metric(
                relevance_basis,
                f"启发式评分：回答与问题的重合度为 {relevance_basis:.4f}",
            ),
        }

    def _basis_metric(self, basis: float, reason: str) -> Dict[str, Any]:
        score = self._basis_to_rubric_score(basis)
        return {
            "score": score,
            "normalized_score": round(score / 5.0, 4),
            "reason": reason,
        }

    def _format_generation_scores(self, parsed_scores: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for metric_name in self.GENERATION_METRICS:
            payload = parsed_scores.get(metric_name, {})
            score = self._sanitize_score(payload.get("score", 1))
            result[metric_name] = {
                "score": score,
                "normalized_score": round(score / 5.0, 4),
                "reason": str(payload.get("reason", "")).strip(),
            }
        return result

    def _summarize_retrieval(self, sample_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        recalls = [item["retrieval_metrics"]["recall"] for item in sample_reports]
        precisions = [item["retrieval_metrics"]["precision"] for item in sample_reports]
        mrrs = [item["retrieval_metrics"]["mrr"] for item in sample_reports]

        return {
            "avg_recall": round(mean(recalls), 4) if recalls else 0.0,
            "avg_precision": round(mean(precisions), 4) if precisions else 0.0,
            "avg_mrr": round(mean(mrrs), 4) if mrrs else 0.0,
        }

    def _summarize_generation(self, sample_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        for metric_name in self.GENERATION_METRICS:
            scores = [item["generation_metrics"][metric_name]["score"] for item in sample_reports]
            normalized_scores = [
                item["generation_metrics"][metric_name]["normalized_score"]
                for item in sample_reports
            ]
            summary[f"avg_{metric_name}"] = {
                "avg_score": round(mean(scores), 4) if scores else 0.0,
                "avg_normalized_score": round(mean(normalized_scores), 4) if normalized_scores else 0.0,
            }

        overall_scores = [item.get("generation_overall_score", 0.0) for item in sample_reports]
        overall_normalized_scores = [
            item.get("generation_overall_normalized_score", 0.0)
            for item in sample_reports
        ]
        summary["avg_overall_generation"] = {
            "avg_score": round(mean(overall_scores), 4) if overall_scores else 0.0,
            "avg_normalized_score": round(mean(overall_normalized_scores), 4)
            if overall_normalized_scores
            else 0.0,
        }
        return summary

    def _build_context_for_judge(
        self,
        docs: List[Document],
        max_docs: int = 3,
        max_chars: int = 3500,
    ) -> str:
        parts: List[str] = []
        current_length = 0

        for index, doc in enumerate(docs[:max_docs], 1):
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            category = doc.metadata.get("category", "未知分类")
            section = f"【证据 {index}】{dish_name} | {category}\n{doc.page_content}\n"
            if current_length + len(section) > max_chars:
                break
            parts.append(section)
            current_length += len(section)

        return "\n".join(parts)

    def _build_reference_answer(self, doc: Document) -> str:
        content = doc.page_content
        ingredients = self._extract_markdown_section(content, "必备原料和工具")
        calculation = self._extract_markdown_section(content, "计算")
        steps = self._extract_markdown_section(content, "操作")

        parts: List[str] = []
        if ingredients:
            parts.append("必备原料和工具:\n" + ingredients[:300])
        if calculation:
            parts.append("食材用量:\n" + calculation[:300])
        if steps:
            parts.append("操作步骤:\n" + steps[:600])
        return "\n\n".join(parts).strip()

    def _get_ranked_dish_names(self, docs: List[Document]) -> List[str]:
        ranked_names: List[str] = []
        seen = set()

        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            normalized = self._normalize_name(dish_name)
            if normalized not in seen:
                ranked_names.append(dish_name)
                seen.add(normalized)

        return ranked_names

    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```json", "", cleaned)
        cleaned = re.sub(r"^```", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

        if cleaned.startswith("{") and cleaned.endswith("}"):
            return json.loads(cleaned)

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"无法从 LLM 输出中解析 JSON: {raw_response}")

    def _basis_to_rubric_score(self, value: float) -> int:
        value = max(0.0, min(1.0, value))
        return max(1, min(5, round(1 + value * 4)))

    def _sanitize_score(self, score: Any) -> int:
        try:
            numeric_score = int(round(float(score)))
        except (TypeError, ValueError):
            numeric_score = 1
        return max(1, min(5, numeric_score))

    def _normalize_name(self, text: str) -> str:
        return re.sub(r"\s+", "", text).lower()

    def _text_similarity(self, left: str, right: str) -> float:
        left_terms = set(self._extract_terms(left))
        right_terms = set(self._extract_terms(right))
        if not left_terms or not right_terms:
            return 0.0

        overlap = len(left_terms.intersection(right_terms))
        union = len(left_terms.union(right_terms))
        return overlap / union if union else 0.0

    def _extract_terms(self, text: str) -> List[str]:
        if not text:
            return []

        compact = re.sub(r"\s+", "", text.lower())
        english_terms = re.findall(r"[a-z0-9_]+", compact)
        chinese_groups = re.findall(r"[\u4e00-\u9fff]+", compact)

        terms = list(english_terms)
        for group in chinese_groups:
            terms.append(group)
            if len(group) >= 2:
                terms.extend(group[index:index + 2] for index in range(len(group) - 1))

        return terms

    def _extract_markdown_section(self, content: str, title: str) -> str:
        pattern = rf"^##\s*{re.escape(title)}\s*$([\s\S]*?)(?=^##\s|\Z)"
        match = re.search(pattern, content, re.MULTILINE)
        return match.group(1).strip() if match else ""
