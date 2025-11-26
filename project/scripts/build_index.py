from __future__ import annotations

import numpy as np
import faiss

from pathlib import Path
from typing import Dict, List, Tuple

from .common import (
    DATA_DIR,
    FAISS_INDEX_PATH,
    IMAGE_VECTOR_DIR,
    INDEX_DIR,
    MAPPING_PATH,
    TEXT_VECTOR_DIR,
    ensure_directories,
    load_json,
    save_json,
)


class FaissIndexManager:
    """负责构建 FAISS IndexFlatIP 索引，并维护 vector_id 对应关系。"""

    def __init__(self) -> None:
        ensure_directories()

    def _load_vectors(self) -> Tuple[List[np.ndarray], List[Dict]]:
        mapping = load_json(MAPPING_PATH)
        vectors: List[np.ndarray] = []
        vector_info: List[Dict] = []

        def _append_vectors(entries: Dict[str, str], chunk_type: str) -> None:
            for ref_id, rel_path in entries.items():
                path = DATA_DIR / rel_path
                if not path.exists():
                    continue
                vec = np.load(path).astype(np.float32)
                vectors.append(vec)
                vector_info.append({"ref_id": ref_id, "type": chunk_type})

        _append_vectors(mapping.get("text", {}), "text")
        _append_vectors(mapping.get("image", {}), "image")
        return vectors, vector_info

    def build(self) -> Dict:
        vectors, info = self._load_vectors()
        if not vectors:
            raise RuntimeError("No vectors available. Run embeddings first.")

        dim = vectors[0].shape[0]
        index = faiss.IndexFlatIP(dim)
        matrix = np.vstack(vectors).astype(np.float32)
        index.add(matrix)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))

        mapping = load_json(MAPPING_PATH)
        mapping["vector_ids"] = [
            {"vector_id": idx, **entry} for idx, entry in enumerate(info)
        ]
        save_json(MAPPING_PATH, mapping)
        return {"vectors": len(vectors), "dim": dim}

