"""
C8 RAG system entrypoint.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
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

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.answer_cache: Dict[str, str] = {}

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        """Initialize modules."""
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
        """Initialize system and knowledge base once."""
        self.initialize_system()
        self.build_knowledge_base()

    def analyze_and_retrieve(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """Run query analysis and retrieval without generation."""
        if not all([self.data_module, self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        self._print(verbose, f"\n用户问题: {question}")

        route_type = self.generation_module.query_router(question)
        self._print(verbose, f"查询类型: {route_type}")

        self._print(verbose, "正在执行查询重写...")
        rewritten_query = self.generation_module.query_rewrite(
            question,
            route_type=route_type,
        )
        self._print(verbose, f"重写后的查询: {rewritten_query}")

        filters = self._extract_filters_from_query(question)
        search_top_k = self._get_search_top_k(route_type)

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

        relevant_docs = (
            self.data_module.get_parent_documents(relevant_chunks)
            if relevant_chunks
            else []
        )

        if relevant_chunks:
            self._print(verbose, f"命中文本块数: {len(relevant_chunks)}")
        else:
            self._print(verbose, "未命中相关文本块。")

        if relevant_docs:
            dish_names = [doc.metadata.get("dish_name", "未知菜品") for doc in relevant_docs]
            self._print(verbose, f"命中文档: {', '.join(dish_names)}")
        elif verbose and relevant_chunks:
            self._print(verbose, "未能还原出完整父文档。")

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
        """Full RAG pipeline."""
        cache_key = self._normalize_question(question)
        cached_answer = self.answer_cache.get(cache_key)
        if cached_answer:
            self._print(True, "\n命中重复问题缓存，直接返回已有答案。")
            return cached_answer

        pipeline = self.analyze_and_retrieve(question, verbose=True)
        result = self.generate_answer(
            question=question,
            route_type=pipeline["route_type"],
            relevant_docs=pipeline["relevant_docs"],
            stream=stream,
            verbose=True,
        )

        if isinstance(result, str):
            self.answer_cache[cache_key] = result
            return result

        return self._cache_streaming_answer(cache_key, result)

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
                    print("\n")
                else:
                    print(f"{result}\n")
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"处理问题时出错: {exc}")

        print("\n已退出食谱 RAG 系统。")

    @staticmethod
    def _print(enabled: bool, message: str):
        if enabled:
            print(message)

    @staticmethod
    def _normalize_question(question: str) -> str:
        return "".join(question.lower().split())

    def _cache_streaming_answer(self, cache_key: str, stream_result: Iterator[str]):
        collected_chunks: List[str] = []
        for chunk in stream_result:
            collected_chunks.append(chunk)
            yield chunk
        self.answer_cache[cache_key] = "".join(collected_chunks)


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
        help="运行 RAG 评估并输出检索/生成质量结果",
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


def main():
    args = parse_args()

    try:
        rag_system = RecipeRAGSystem()

        if args.interactive or not (args.generate_eval_dataset or args.evaluate):
            rag_system.run_interactive()
            return

        rag_system.setup()
        evaluator = RAGEvaluator(
            rag_system,
            use_llm_judge=not args.no_llm_judge,
        )

        if args.generate_eval_dataset:
            evaluator.generate_default_dataset(
                output_path=args.dataset,
                sample_size=args.sample_size,
            )
            print(f"评估数据集已生成: {args.dataset}")

        if args.evaluate:
            report = evaluator.evaluate_dataset(
                dataset_path=args.dataset,
                output_path=args.output,
                sample_limit=args.sample_limit,
            )
            evaluator.print_summary(report)
            print(f"\n评估报告已保存: {args.output}")

    except Exception as exc:
        logger.error(f"系统运行出错: {exc}")
        print(f"系统错误: {exc}")


if __name__ == "__main__":
    main()
