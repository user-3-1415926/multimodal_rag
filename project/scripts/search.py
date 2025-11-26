from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .common import (
    DATA_DIR,
    FAISS_INDEX_PATH,
    MANIFEST_PATH,
    MAPPING_PATH,
    ensure_directories,
    load_json,
)
from .embed import Embedder


def _load_mapping_and_manifest() -> Tuple[Dict, Dict]:
    mapping = load_json(MAPPING_PATH)
    manifest = load_json(MANIFEST_PATH)
    if not mapping or "vector_ids" not in mapping:
        raise RuntimeError("mapping.json missing vector_ids. Rebuild index first.")
    return mapping, manifest


def _vector_id_to_ref(mapping: Dict, vector_id: int) -> Tuple[str, str]:
    for entry in mapping.get("vector_ids", []):
        if entry["vector_id"] == vector_id:
            return entry["ref_id"], entry["type"]
    raise KeyError(f"vector_id {vector_id} missing from mapping.")


def search(query_text: str, top_k: int = 10) -> List[Dict]:
    """
    1. Embed query text
    2. Search FAISS
    3. Map vector IDs back to items
    4. Aggregate scores per item
    5. Return metadata + top chunks
    """
    ensure_directories()
    mapping, manifest = _load_mapping_and_manifest()
    embedder = Embedder()

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index missing. Run build_index.py first.")
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    query_vector = embedder.embed_text(query_text).astype(np.float32)[None, :]
    scores, ids = index.search(query_vector, top_k)

    aggregated = defaultdict(lambda: {"score": 0.0, "chunks": []})
    results = []
    for score, vector_id in zip(scores[0], ids[0]):
        if vector_id < 0:
            continue
        ref_id, ref_type = _vector_id_to_ref(mapping, int(vector_id))
        parent_id = ref_id.split("-")[0] if ref_type == "text" else ref_id
        aggregated[parent_id]["score"] += float(score)
        aggregated[parent_id]["chunks"].append({"id": ref_id, "type": ref_type, "score": float(score)})

    for item_id, payload in sorted(aggregated.items(), key=lambda x: x[1]["score"], reverse=True):
        manifest_entry = manifest.get(item_id, {})
        results.append(
            {
                "item_id": item_id,
                "score": payload["score"],
                "metadata": manifest_entry,
                "chunks": payload["chunks"],
            }
        )
    return results

