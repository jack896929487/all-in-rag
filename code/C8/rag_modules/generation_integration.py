"""
Generation module for C8.
"""

import logging
import os
from typing import List, Optional

from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """LLM integration and answer generation."""

    def __init__(
        self,
        model_name: str = "kimi-k2-0711-preview",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """Initialize the LLM client."""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key,
        )
        logger.info("LLM初始化完成")

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate a normal grounded answer."""
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的食谱问答助手。请严格基于提供的食谱信息回答问题。

用户问题:
{question}

食谱证据:
{context}

要求:
1. 优先直接回答用户问题。
2. 只能依据证据作答，不要编造菜名、食材或做法。
3. 如果证据不足，明确说明局限。
4. 语言简洁、实用。

回答:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": query, "context": context})

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate a detailed step-by-step answer."""
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的烹饪导师。请基于给定食谱信息，为用户生成清晰、可执行的回答。

用户问题:
{question}

食谱证据:
{context}

要求:
1. 先简短说明推荐或菜品概况。
2. 如果问题涉及做法，按步骤说明。
3. 如果问题涉及食材，明确列出关键食材。
4. 不要编造证据中没有的信息。
5. 如果信息不完整，直接说明。

回答:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": query, "context": context})

    def query_rewrite(self, query: str, route_type: Optional[str] = None) -> str:
        """Rewrite the user query into a retrieval-friendly query."""
        prompt = ChatPromptTemplate.from_template(
            """
你是一个 RAG 检索查询改写助手。你的目标不是回答问题，而是把用户问题改写成更适合食谱库检索的查询。

输入问题:
{query}

路由类型:
{route_type}

改写要求:
1. 保留用户原始意图，不要改变约束。
2. 明确保留这些信息：菜品目标、用餐场景、健康目标、营养目标、口味偏好、难度、烹饪方式。
3. 如果是推荐类问题，可以补充有助于检索的关键词，例如“高蛋白、健康、低油、午餐、中饭、增肌、鸡蛋、鸡胸肉、牛肉、豆腐”等，但不要编造具体菜名。
4. 如果原问题已经很适合检索，可以只做轻微整理。
5. 输出必须是单行中文检索查询，不要解释。

改写结果:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke(
            {
                "query": query,
                "route_type": route_type or "unknown",
            }
        ).strip()

        if rewritten_query != query:
            logger.info(f"查询已重写: '{query}' -> '{rewritten_query}'")
        else:
            logger.info(f"查询无需重写: '{query}'")
        return rewritten_query

    def query_router(self, query: str) -> str:
        """Route query into list/detail/general."""
        prompt = ChatPromptTemplate.from_template(
            """
请判断用户问题属于以下哪一类，只输出一个标签:

1. list
适用于用户要推荐、菜品候选、吃什么、搭配什么。

2. detail
适用于用户明确要做法、步骤、食材、制作细节。

3. general
适用于一般知识、原理、营养或烹饪常识。

用户问题:
{query}

输出:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()
        if result in {"list", "detail", "general"}:
            return result
        return "general"

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate recommendation-style answer with LLM."""
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的营养与食谱助手。请基于给定食谱证据，为用户做推荐。

用户问题:
{question}

候选食谱证据:
{context}

要求:
1. 只能从证据中出现的菜品里推荐，不要编造新菜品。
2. 推荐时优先考虑用户约束，例如健康、蛋白质、低油、增肌、早餐/午餐/晚餐等。
3. 按优先级给出 2-4 个推荐项。
4. 每个推荐项都要给出简短理由。
5. 如果这些候选并不能很好满足用户要求，要明确指出，并说明哪个是相对更合适的选择。

回答:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"question": query, "context": context})
        except Exception as exc:
            logger.warning(f"LLM 推荐生成失败，回退到列表模式: {exc}")
            return self._fallback_list_answer(context_docs)

    def generate_list_answer_stream(self, query: str, context_docs: List[Document]):
        """Generate recommendation-style answer with streaming."""
        if not context_docs:
            yield "抱歉，没有找到相关的菜品信息。"
            return

        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的营养与食谱助手。请基于给定食谱证据，为用户做推荐。

用户问题:
{question}

候选食谱证据:
{context}

要求:
1. 只能从证据中出现的菜品里推荐，不要编造新菜品。
2. 推荐时优先考虑用户约束，例如健康、蛋白质、低油、增肌、早餐/午餐/晚餐等。
3. 按优先级给出 2-4 个推荐项。
4. 每个推荐项都要给出简短理由。
5. 如果这些候选并不能很好满足用户要求，要明确指出，并说明哪个是相对更合适的选择。

回答:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        try:
            for chunk in chain.stream({"question": query, "context": context}):
                yield chunk
        except Exception as exc:
            logger.warning(f"LLM 流式推荐生成失败，回退到列表模式: {exc}")
            yield self._fallback_list_answer(context_docs)

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """Generate a normal grounded answer with streaming."""
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的食谱问答助手。请严格基于提供的食谱信息回答问题。

用户问题:
{question}

食谱证据:
{context}

要求:
1. 优先直接回答用户问题。
2. 只能依据证据作答，不要编造菜名、食材或做法。
3. 如果证据不足，明确说明局限。
4. 语言简洁、实用。

回答:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"question": query, "context": context}):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """Generate a detailed step-by-step answer with streaming."""
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的烹饪导师。请基于给定食谱信息，为用户生成清晰、可执行的回答。

用户问题:
{question}

食谱证据:
{context}

要求:
1. 先简短说明推荐或菜品概况。
2. 如果问题涉及做法，按步骤说明。
3. 如果问题涉及食材，明确列出关键食材。
4. 不要编造证据中没有的信息。
5. 如果信息不完整，直接说明。

回答:
""".strip()
        )

        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"question": query, "context": context}):
            yield chunk

    def _fallback_list_answer(self, context_docs: List[Document]) -> str:
        dish_names: List[str] = []
        for doc in context_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        if not dish_names:
            return "抱歉，没有找到相关的菜品信息。"

        return "可参考这些候选菜品：\n" + "\n".join(
            f"{index + 1}. {dish_name}" for index, dish_name in enumerate(dish_names[:4])
        )

    def _build_context(self, docs: List[Document], max_length: int = 2800) -> str:
        """Build context string from retrieved documents."""
        if not docs:
            return "暂无相关食谱信息。"

        context_parts = []
        current_length = 0

        for index, doc in enumerate(docs, 1):
            metadata_parts = [f"【食谱 {index}】"]
            if "dish_name" in doc.metadata:
                metadata_parts.append(doc.metadata["dish_name"])
            if "category" in doc.metadata:
                metadata_parts.append(f"分类: {doc.metadata['category']}")
            if "difficulty" in doc.metadata:
                metadata_parts.append(f"难度: {doc.metadata['difficulty']}")

            doc_text = " | ".join(metadata_parts) + "\n" + doc.page_content + "\n"
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n" + "=" * 50 + "\n".join(context_parts)
