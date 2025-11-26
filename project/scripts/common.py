from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_TEXT_DIR = RAW_DIR / "texts"
RAW_IMAGE_DIR = RAW_DIR / "images"
RAW_METADATA_DIR = RAW_DIR / "metadata"
CHUNK_DIR = DATA_DIR / "chunks"
VECTORS_DIR = DATA_DIR / "vectors"
TEXT_VECTOR_DIR = VECTORS_DIR / "text"
IMAGE_VECTOR_DIR = VECTORS_DIR / "image"
INDEX_DIR = VECTORS_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
MAPPING_PATH = INDEX_DIR / "mapping.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"


REQUIRED_DIRS = [
    RAW_TEXT_DIR,
    RAW_IMAGE_DIR,
    RAW_METADATA_DIR,
    CHUNK_DIR,
    TEXT_VECTOR_DIR,
    IMAGE_VECTOR_DIR,
    INDEX_DIR,
]


def ensure_directories() -> None:
    """Create every required directory in the pipeline."""
    for path in REQUIRED_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

