# multi_retrieval.py

import re
from typing import Dict, List, Tuple, Union

import numpy as np
from langchain_community.vectorstores import FAISS

from multi_embedding import CLIPEmbeddingsWrapper, load_clip_model

try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# 确保您可以从 multi_embedding 导入 load_clip_model 和 CLIPEmbeddingsWrapper

def load_multimodal_index(pthName: str, idxName: str):
    """
    加载多模态 FAISS 向量索引和相应的 CLIP 模型。
    """
    # 1. 加载 CLIP 模型和封装器
    clip_model, clip_processor = load_clip_model()
    clip_embeddings = CLIPEmbeddingsWrapper(clip_model, clip_processor)

    # 2. 加载 FAISS 向量存储
    print(f"Loading multimodal index from {pthName}/{idxName}...")
    try:
        loaded_store = FAISS.load_local(
            folder_path=pthName, 
            index_name=idxName, 
            embeddings=clip_embeddings, 
            allow_dangerous_deserialization=True 
        )
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise
        
    print("Multimodal vector store loaded successfully.")
    return loaded_store


CHUNK_WEIGHTS = {
    "category": 1.2,
    "date": 1.1,
    "description": 1.0,
    "image": 1.3,
}

CATEGORY_KEYWORDS = ["背包", "包", "钥匙", "手机", "卡", "水杯", "杯", "书", "雨伞", "眼镜", "耳机", "钱包", "证件"]
DATE_PATTERN = r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日[^\n。]*"
WEEKDAY_PATTERN = r"周[一二三四五六日天][^\n。]*"
TIME_WORDS = ["上午", "下午", "晚上", "中午", "傍晚", "凌晨"]


def _distance_to_cosine_similarity(distance: float) -> float:
    """
    将 FAISS L2 距离转换为余弦相似度。
    
    对于已归一化的向量（CLIP 向量通常已归一化），L2 距离 d 和余弦相似度 sim 的关系为：
    d^2 = 2 * (1 - sim)
    因此：sim = 1 - d^2 / 2
    
    参数：distance - FAISS 返回的 L2 距离
    返回：余弦相似度，范围 [0, 1]（对于已归一化向量）
    """
    similarity = 1 - (distance ** 2) / 2
    similarity = min(1.0, similarity)
    return max(-1.0, similarity)


def _infer_category_from_query(query: str) -> str:
    for keyword in CATEGORY_KEYWORDS:
        if keyword in query:
            return keyword
    # 默认返回前 10~12 个字符作为类别描述
    cleaned = query.strip()
    return cleaned[:12] if cleaned else ""


def _extract_date_phrase(query: str) -> str:
    match = re.search(DATE_PATTERN, query)
    if match:
        return match.group(0)
    match = re.search(WEEKDAY_PATTERN, query)
    if match:
        return match.group(0)
    for word in TIME_WORDS:
        if word in query:
            return word
    return ""


def _split_query_into_chunks(query: str) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    category = _infer_category_from_query(query)
    if category:
        chunks.append(("category", category))
    date_phrase = _extract_date_phrase(query)
    if date_phrase:
        chunks.append(("date", date_phrase))
    description = query.strip()
    if description:
        chunks.append(("description", description))
    # 去重保持顺序
    seen = set()
    filtered = []
    for chunk_type, text in chunks:
        key = (chunk_type, text)
        if key in seen:
            continue
        seen.add(key)
        filtered.append((chunk_type, text))
    return filtered


def _init_item_entry(item_scores: Dict[str, dict], item_id: str) -> dict:
    if item_id not in item_scores:
        item_scores[item_id] = {"score": 0.0, "hits": [], "item_id": item_id}
    return item_scores[item_id]


def _record_hit(
    item_scores: Dict[str, dict],
    doc: Document,
    raw_score: float,
    adjusted_score: float,
    query_chunk: str,
) -> None:
    item_id = doc.metadata.get("item_id", "unknown")
    entry = _init_item_entry(item_scores, item_id)
    entry["score"] += adjusted_score
    entry["hits"].append(
        {
            "chunk_type": doc.metadata.get("chunk_type"),
            "query_chunk": query_chunk,
            "raw_score": raw_score,
            "adjusted_score": adjusted_score,
            "source_file": doc.metadata.get("source_file"),
            "chunk_text": doc.metadata.get("chunk_text"),
        }
    )


def _normalize_retrieved_scores(
    retrieved_with_scores: List[Tuple[Document, float]]
) -> List[Tuple[Document, float]]:
    if not retrieved_with_scores:
        return []
    raw_scores = [float(s) for _, s in retrieved_with_scores]
    max_raw = max(raw_scores)
    min_raw = min(raw_scores)
    if -1.0 <= min_raw and max_raw <= 1.0:
        return [(doc, float(score)) for doc, score in retrieved_with_scores]
    return [
        (doc, _distance_to_cosine_similarity(float(score)))
        for doc, score in retrieved_with_scores
    ]


def _search_with_scores(vector_store: FAISS, query: str, k: int) -> List[Tuple[Document, float]]:
    try:
        raw_results = vector_store.similarity_search_with_score(query, k=k)
        return _normalize_retrieved_scores(raw_results)
    except Exception:
        fallback_docs = vector_store.similarity_search(query, k=k)
        return [(doc, 0.5) for doc in fallback_docs]


def _search_with_scores_by_vector(
    vector_store: FAISS, query_vector: List[float], k: int
) -> List[Tuple[Document, float]]:
    try:
        raw_results = vector_store.similarity_search_by_vector_with_score(query_vector, k=k)
        return _normalize_retrieved_scores(raw_results)
    except Exception:
        fallback_docs = vector_store.similarity_search_by_vector(query_vector, k=k)
        return [(doc, 0.5) for doc in fallback_docs]


def multi_search(
    query: str, vector_store: FAISS, k: int = 4
) -> Tuple[List[Tuple[Document, float]], Dict[str, dict]]:
    """
    基于语义块的多模态搜索。
    返回： (chunk_hits, aggregated_scores_by_item)
    """
    query_chunks = _split_query_into_chunks(query)
    if not query_chunks:
        query_chunks = [("description", query)]

    chunk_hits: List[Tuple[Document, float]] = []
    item_scores: Dict[str, dict] = {}

    for chunk_type, chunk_text in query_chunks:
        base_results = _search_with_scores(vector_store, chunk_text, k)
        weight = CHUNK_WEIGHTS.get(chunk_type, 1.0)
        for doc, base_score in base_results:
            adjusted = base_score * weight
            chunk_hits.append((doc, adjusted))
            _record_hit(item_scores, doc, base_score, adjusted, chunk_type)

    return chunk_hits, item_scores


def multi_search_image(
    image_path: str, vector_store: FAISS, k: int = 4
) -> Tuple[List[Tuple[Document, float]], Dict[str, dict]]:
    """
    执行基于图像的多模态搜索，返回图像块命中和按 item_id 聚合得分。
    """
    print(f"Performing image search for: {image_path}...")

    clip_embeddings: CLIPEmbeddingsWrapper = vector_store.embeddings
    query_vector = clip_embeddings.embed_image(image_path)

    if all(v == 0.0 for v in query_vector):
        print("Warning: Image embedding failed, cannot perform search.")
        return [], {}

    base_results = _search_with_scores_by_vector(vector_store, query_vector, k)
    chunk_hits: List[Tuple[Document, float]] = []
    item_scores: Dict[str, dict] = {}
    weight = CHUNK_WEIGHTS.get("image", 1.3)

    for doc, base_score in base_results:
        adjusted = base_score * weight
        chunk_hits.append((doc, adjusted))
        _record_hit(item_scores, doc, base_score, adjusted, "image")

    return chunk_hits, item_scores
