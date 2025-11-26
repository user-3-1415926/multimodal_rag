from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .common import IMAGE_VECTOR_DIR, TEXT_VECTOR_DIR, ensure_directories


class Embedder:
    """CLIP embedding helper for both text chunks and images."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        ensure_directories()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (vector / norm).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return self._normalize(features.squeeze(0).detach().cpu().numpy())

    def embed_image_file(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return self._normalize(features.squeeze(0).detach().cpu().numpy())

    def embed_text_chunks(self, chunk_records: List[Dict]) -> Dict[str, Path]:
        vector_paths: Dict[str, Path] = {}
        for record in chunk_records:
            chunk_id = record["chunk_id"]
            vector = self.embed_text(record["text"])
            out_path = TEXT_VECTOR_DIR / f"{chunk_id}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, vector)
            vector_paths[chunk_id] = out_path
        return vector_paths

    def embed_image(self, item_id: str, image_path: Path) -> Path:
        vector = self.embed_image_file(image_path)
        out_path = IMAGE_VECTOR_DIR / f"{item_id}.npy"
        np.save(out_path, vector)
        return out_path

