# multi_generation.py

from typing import Dict, List, Tuple, Union
import numpy as np
try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}
try:
    from ollama import generate  # 假设您使用 Ollama
except Exception:
    # 回退 stub：如果本地没有安装 `ollama` 包，则使用一个简单的替代函数，
    # 使脚本能继续运行并返回可读的占位响应。
    def generate(model: str, prompt: str):
        return {"response": f"[stub] Ollama 客户端不可用，已收到提示 (model={model})."}
from langchain_community.vectorstores import FAISS


def normalize_scores_to_percentage(scores: List[float]) -> List[float]:
    """
    将分数线性映射到 0-100 区间，用于展示。
    """
    if not scores:
        return []
    max_val = max(scores)
    min_val = min(scores)
    if max_val == min_val:
        if max_val > 0:
            uniform = 100.0
        elif max_val < 0:
            uniform = 0.0
        else:
            uniform = 0.0
        return [uniform for _ in scores]
    span = max_val - min_val
    return [((s - min_val) / span) * 100.0 for s in scores]


def format_multimodal_context(
    retrieved_docs: List[Union[Document, Tuple[Document, float]]]
) -> tuple[str, str, List[float], List[float]]:
    """
    格式化检索到的多模态文档，分离文本内容和图像信息，并返回原始分数与 0-100 展示值。
    返回：(文本上下文, 图像路径, 原始分数列表, 归一展示列表)
    """
    text_context = []
    image_paths = []
    scores: List[float] = []
    
    for item in retrieved_docs:
        # 支持两种格式：Document 或 (Document, score)
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
        else:
            doc, score = item, None

        if score is None and doc and hasattr(doc, "metadata"):
            score = doc.metadata.get("similarity") or doc.metadata.get("score")

        # 如果还是没有分数，使用默认值 0.5
        if score is None:
            score = 0.5
        score = float(score)
        scores.append(score)

        doc_type = doc.metadata.get("type") if doc and hasattr(doc, 'metadata') else None
        if doc_type == "text":
            src = doc.metadata.get('source', 'unknown')
            text_context.append(f"--- 文本块 (Source: {src}, score={score:.4f}) ---\n{doc.page_content}")
        elif doc_type == "image":
            image_paths.append(f"{doc.page_content} (score={score:.4f})")
    
    # 将所有文本块合并为一个字符串
    combined_text = "\n\n".join(text_context)
    # 将所有检索到的图像路径合并为一个字符串
    combined_images = "\n".join(image_paths)

    normalized_scores = normalize_scores_to_percentage(scores)
    return combined_text, combined_images, scores, normalized_scores


def _aggregate_scores_by_item(
    retrieved_docs: List[Union[Document, Tuple[Document, float]]]
) -> Dict[str, dict]:
    aggregated: Dict[str, dict] = {}
    for item in retrieved_docs:
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc, score = item, 0.5
        item_id = doc.metadata.get("item_id", "unknown")
        if item_id not in aggregated:
            aggregated[item_id] = {"item_id": item_id, "score": 0.0, "hits": []}
        aggregated[item_id]["score"] += float(score)
        aggregated[item_id]["hits"].append(
            {
                "chunk_type": doc.metadata.get("chunk_type"),
                "query_chunk": "unknown",
                "raw_score": float(score),
                "adjusted_score": float(score),
                "source_file": doc.metadata.get("source_file"),
                "chunk_text": doc.metadata.get("chunk_text"),
            }
        )
    return aggregated


def multi_rag_fct(
    query: str,
    llmModelName: str,
    vectorStore: FAISS,
    retrieved_docs: List[Union[Document, Tuple[Document, float]]] | None = None,
    aggregated_hits: Dict[str, dict] | None = None,
):
    """
    简洁版多模态 RAG：检索 → 生成。
    """
    from multi_retrieval import multi_search

    # 1. 检索
    if not retrieved_docs:
        retrieved_docs, aggregated_hits = multi_search(query, vectorStore, k=4)
    if aggregated_hits is None:
        aggregated_hits = _aggregate_scores_by_item(retrieved_docs)

    # 2. 上下文格式化
    text_context, image_paths, scores_list, normalized_scores = format_multimodal_context(
        retrieved_docs
    )

    # 分数列表
    score_lines = [
        f"{raw:.6f} | {scaled:.2f}"
        for raw, scaled in zip(scores_list, normalized_scores)
    ]
    scores_str = "\n".join(score_lines)

    # Item 综合得分
    item_rankings = sorted(
        aggregated_hits.values(), key=lambda x: x["score"], reverse=True
    )
    item_summary = "\n".join(
        f"Item {it['item_id']} | 综合={it['score']:.4f}"
        for it in item_rankings
    )

    # 3. Prompt
    augmented_prompt = f"""
你是一名检索增强问答助手，只能根据以下内容回答，不得使用外部知识。

### 文本上下文
{text_context}

### 图像路径
{image_paths}

### 向量分数（raw | 0-100）
{scores_str}

### Item 综合得分
{item_summary}

### 用户问题
{query}

### 输出要求
请对检索到的每个项目给出：
- 匹配度 (0–100)
- 简要理由（只根据上下文）
最后总结哪个 item 最可能匹配，并说明原因。
"""

    # ★★ 返回 main_rag 期望的 6 个值 ★★
    return (
        text_context
        ,image_paths
        ,scores_list
        ,normalized_scores
        ,item_rankings
        ,generate(llmModelName, augmented_prompt).get("response", "")
    )
