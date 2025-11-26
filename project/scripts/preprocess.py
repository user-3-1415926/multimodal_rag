from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .common import (
    CHUNK_DIR,
    DATA_DIR,
    IMAGE_VECTOR_DIR,
    MANIFEST_PATH,
    MAPPING_PATH,
    RAW_IMAGE_DIR,
    RAW_METADATA_DIR,
    RAW_TEXT_DIR,
    TEXT_VECTOR_DIR,
    ensure_directories,
    load_json,
    save_json,
)
from .chunk_text import chunk_text
from .embed import Embedder


@dataclass
class RawDataPaths:
    text_path: Path
    image_path: Path
    metadata_path: Path


class RawDataManager:
    """负责落地保存原始文本、图片与元数据。"""

    def __init__(self) -> None:
        ensure_directories()

    def save_text(self, item_id: str, text: str) -> Path:
        path = RAW_TEXT_DIR / f"{item_id}.txt"
        path.write_text(text.strip(), encoding="utf-8")
        return path

    def save_image(self, item_id: str, image_path: str) -> Path:
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        dst = RAW_IMAGE_DIR / f"{item_id}{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        return dst

    def save_metadata(
        self,
        item_id: str,
        category: str,
        location: str,
        date: str,
        source: str,
        image_path: Path,
        text_path: Path,
    ) -> Path:
        metadata = {
            "id": item_id,
            "category": category,
            "location": location,
            "date": date,
            "source": source,
            "image_path": str(image_path.relative_to(DATA_DIR)),
            "text_path": str(text_path.relative_to(DATA_DIR)),
        }
        meta_path = RAW_METADATA_DIR / f"{item_id}.json"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def save_all(
        self,
        item_id: str,
        text: str,
        image_path: str,
        category: str,
        location: str,
        date: str,
        source: str = "user_upload",
    ) -> RawDataPaths:
        ensure_directories()
        text_fp = self.save_text(item_id, text)
        image_fp = self.save_image(item_id, image_path)
        metadata_fp = self.save_metadata(
            item_id, category, location, date, source, image_fp, text_fp
        )
        return RawDataPaths(text_fp, image_fp, metadata_fp)


class ManifestManager:
    """记录所有条目的清单（manifest.json）。"""

    def __init__(self) -> None:
        ensure_directories()
        self.manifest: Dict = load_json(MANIFEST_PATH)

    def update_item(
        self,
        item_id: str,
        text_file: Path,
        image_file: Path,
        metadata_file: Path,
        chunks: List[str],
        vector_paths: Dict[str, Dict[str, str]],
    ) -> None:
        self.manifest.setdefault(item_id, {})
        self.manifest[item_id].update(
            {
                "text_file": str(text_file.relative_to(DATA_DIR)),
                "image_file": str(image_file.relative_to(DATA_DIR)),
                "metadata": str(metadata_file.relative_to(DATA_DIR)),
                "chunks": chunks,
                "vectors": vector_paths,
            }
        )
        save_json(MANIFEST_PATH, self.manifest)


def _update_mapping(
    text_vectors: Dict[str, Path],
    image_vectors: Optional[Dict[str, Path]] = None,
) -> None:
    mapping = load_json(MAPPING_PATH)
    mapping.setdefault("text", {})
    mapping.setdefault("image", {})
    for chunk_id, path in text_vectors.items():
        mapping["text"][chunk_id] = str(path.relative_to(DATA_DIR))
    if image_vectors:
        for item_id, path in image_vectors.items():
            mapping["image"][item_id] = str(path.relative_to(DATA_DIR))
    save_json(MAPPING_PATH, mapping)


def ingest_item(
    item_id: str,
    text: str,
    image_path: str,
    category: str,
    location: str,
    date: str,
) -> Dict:
    """
    End-to-end ingestion pipeline:
      1. Save raw inputs
      2. Chunk text
      3. Embed text/image
      4. Update mapping.json
      5. Update manifest.json
    """
    ensure_directories()
    raw_manager = RawDataManager()
    manifest = ManifestManager()
    embedder = Embedder()

    # Step 1: persist raw data
    raw_paths = raw_manager.save_all(
        item_id=item_id,
        text=text,
        image_path=image_path,
        category=category,
        location=location,
        date=date,
    )

    # Step 2: chunk text into semantic units
    chunk_records = chunk_text(item_id)
    chunk_ids = [chunk["chunk_id"] for chunk in chunk_records]

    # Step 3: embeddings
    text_vector_paths = embedder.embed_text_chunks(chunk_records)
    image_vector_path = embedder.embed_image(item_id, raw_paths.image_path)

    # Step 4: mapping update
    _update_mapping(
        text_vectors=text_vector_paths,
        image_vectors={item_id: image_vector_path} if image_vector_path else None,
    )

    # Step 5: manifest update
    vector_info = {
        "image": str(image_vector_path.relative_to(DATA_DIR)) if image_vector_path else "",
        "text_chunks": {cid: str(path.relative_to(DATA_DIR)) for cid, path in text_vector_paths.items()},
    }
    manifest.update_item(
        item_id,
        text_file=raw_paths.text_path,
        image_file=raw_paths.image_path,
        metadata_file=raw_paths.metadata_path,
        chunks=chunk_ids,
        vector_paths=vector_info,
    )

    return {
        "item_id": item_id,
        "chunks": chunk_records,
        "text_vectors": {k: str(v) for k, v in text_vector_paths.items()},
        "image_vector": str(image_vector_path) if image_vector_path else "",
    }

