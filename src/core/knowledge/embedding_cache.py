from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from .cwe_knowledge_base import CWEEntry
from .embedding_text import build_embedding_text

FloatMatrix = npt.NDArray[np.float32]
IntVector = npt.NDArray[np.int64]
EncodeFn = Callable[[list[str]], npt.NDArray[np.floating]]

_VECTORS_FN = "vectors.npy"
_IDS_FN = "cwe_ids.npy"
_META_FN = "meta.json"

logger = logging.getLogger(name=__name__)


def _fingerprint(docs: list[CWEEntry], model_name: str) -> str:
    payload = {
        "model": model_name,
        "texts": [
            (e.cwe_id, build_embedding_text(e))
            for e in sorted(docs, key=lambda x: x.cwe_id)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_valid(dirpath: Path, fp: str) -> bool:
    meta_path: Path = dirpath / _META_FN
    if not (
        meta_path.is_file()
        and (dirpath / _VECTORS_FN).is_file()
        and (dirpath / _IDS_FN).is_file()
    ):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return meta.get("fingerprint") == fp


def _load(dirpath: Path) -> tuple[FloatMatrix, IntVector]:
    vectors: FloatMatrix = np.load(dirpath / _VECTORS_FN).astype(np.float32, copy=False)
    ids: IntVector = np.load(dirpath / _IDS_FN).astype(np.int64, copy=False)
    if vectors.shape[0] != ids.shape[0]:
        raise ValueError(f"Cache corrupt: vectors {vectors.shape} vs ids {ids.shape}")
    return vectors, ids


def _save(
    dirpath: Path,
    vectors: FloatMatrix,
    ids: IntVector,
    fp: str,
    model_name: str,
) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    np.save(dirpath / _VECTORS_FN, vectors)
    np.save(dirpath / _IDS_FN, ids)
    meta = {
        "model": model_name,
        "fingerprint": fp,
        "n": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
    }
    (dirpath / _META_FN).write_text(json.dumps(meta, indent=2))

    logger.info(f"Embeddings successfully cached in {dirpath}")


def load_or_build(
    docs: list[CWEEntry],
    model_name: str,
    cache_dir: Path,
    encode_fn: EncodeFn,
) -> tuple[FloatMatrix, IntVector]:
    fp = _fingerprint(docs, model_name)
    if _cache_valid(cache_dir, fp):
        return _load(cache_dir)

    logging.warning("Fingerprint mismatch -> Start cache embeddings rebuild")
    ordered: list[CWEEntry] = sorted(docs, key=lambda e: e.cwe_id)
    texts: list[str] = [build_embedding_text(e) for e in ordered]
    raw: npt.NDArray[np.floating] = encode_fn(texts)
    vectors: FloatMatrix = np.ascontiguousarray(raw, dtype=np.float32)
    ids: IntVector = np.array([e.cwe_id for e in ordered], dtype=np.int64)

    if vectors.ndim != 2 or vectors.shape[0] != len(ordered):
        raise ValueError(
            f"encode_fn returned shape {vectors.shape}, expected ({len(ordered)}, D)"
        )

    _save(cache_dir, vectors, ids, fp, model_name)
    return vectors, ids
