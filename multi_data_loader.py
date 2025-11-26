# multi_data_loader.py

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    from langchain.schema import Document
except Exception:
    # 简单回退 Document 定义，仅包含脚本中使用的字段
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}


@dataclass
class ItemRecord:
    """结构化表示单个失物条目，方便生成语义块。"""
    item_id: str
    source_file: str
    raw_text: str
    category: str
    date_found: str
    description: str
    image_paths: List[str]


CATEGORY_PATTERN = re.compile(r"(?:类别|分类|种类|Category)[:：]\s*(.+)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日[^\n。]*")
WEEKDAY_PATTERN = re.compile(r"周[一二三四五六日天][^\n。]*")
TIME_PATTERN = re.compile(r"(上午|中午|下午|傍晚|晚上|凌晨).{0,6}")
IMAGE_PATTERN = re.compile(r"(?:图片|photo|image)[^\n:：]*[:：]\s*([^\s，,；;]+)", re.IGNORECASE)


def _read_file_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[WARN] Text file not found: {file_path}")
        return ""


def _extract_category(text: str) -> str:
    match = CATEGORY_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    first_sentence = re.split(r"[。！？\n]", text, maxsplit=1)[0]
    if len(first_sentence) <= 20 and first_sentence:
        return first_sentence.strip()
    return first_sentence[:20].strip() if first_sentence else "未知物品"


def _extract_date(text: str) -> str:
    for pattern in (DATE_PATTERN, WEEKDAY_PATTERN, TIME_PATTERN):
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    # 尝试匹配“找到时间”或“丢失时间”关键字
    match = re.search(r"(找到|丢失|拾到)[^\n：:]*[:：]\s*([^\n]+)", text)
    if match:
        return match.group(0).strip()
    return ""


def _extract_images(text: str, data_dir: str) -> List[str]:
    images = []
    for match in IMAGE_PATTERN.finditer(text):
        raw_path = match.group(1).strip().strip("。.")
        if not raw_path:
            continue
        abs_path = Path(data_dir) / raw_path
        if abs_path.exists():
            images.append(str(abs_path))
        else:
            images.append(raw_path)
    return images


def _extract_description(text: str, category: str, date_found: str) -> str:
    description = text
    for snippet in (category, date_found):
        if snippet and snippet in description:
            description = description.replace(snippet, "")
    description = IMAGE_PATTERN.sub("", description)
    description = re.sub(r"\s+", " ", description).strip()
    return description if description else text


def _parse_item_file(file_path: str, data_dir: str) -> ItemRecord | None:
    raw_text = _read_file_text(file_path)
    if not raw_text:
        return None

    category = _extract_category(raw_text)
    date_found = _extract_date(raw_text)
    description = _extract_description(raw_text, category, date_found)
    images = _extract_images(raw_text, data_dir)

    return ItemRecord(
        item_id=Path(file_path).stem,
        source_file=file_path,
        raw_text=raw_text,
        category=category or "未知物品",
        date_found=date_found or "时间未知",
        description=description,
        image_paths=images,
    )


def _build_chunk_documents(item: ItemRecord) -> List[Document]:
    docs: List[Document] = []
    base_metadata = {
        "item_id": item.item_id,
        "source_file": item.source_file,
        "timestamp": item.date_found,
    }

    docs.append(
        Document(
            page_content=item.category,
            metadata={
                **base_metadata,
                "chunk_type": "category",
                "chunk_text": item.category,
                "type": "text",
            },
        )
    )

    docs.append(
        Document(
            page_content=item.date_found,
            metadata={
                **base_metadata,
                "chunk_type": "date",
                "chunk_text": item.date_found,
                "type": "text",
            },
        )
    )

    docs.append(
        Document(
            page_content=item.description,
            metadata={
                **base_metadata,
                "chunk_type": "description",
                "chunk_text": item.description,
                "type": "text",
            },
        )
    )

    for image_path in item.image_paths:
        docs.append(
            Document(
                page_content=image_path,
                metadata={
                    **base_metadata,
                    "chunk_type": "image",
                    "chunk_text": item.description,
                    "type": "image",
                    "filename": image_path,
                },
            )
        )
    return docs


def multi_chunk_associate(data_dir: str) -> list[Document]:
    """
    构建多模态语义块：
      - category: 高层分类
      - date: 时间或地点描述
      - description: 细节描述
      - image: CLIP 图像嵌入
    每个块都会携带 item_id/timestamp/filename 元数据，便于下游聚合。
    """
    all_documents: List[Document] = []

    text_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(data_dir)
        for f in files
        if f.lower().endswith((".txt", ".md", ".html", ".htm"))
    ]

    for file_path in text_files:
        item = _parse_item_file(file_path, data_dir)
        if not item:
            continue
        all_documents.extend(_build_chunk_documents(item))

    print(
        f"Loaded {len(text_files)} structured text items -> {len(all_documents)} semantic chunks."
    )
    return all_documents