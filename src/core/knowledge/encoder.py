import numpy as np
from typing import Iterable
from sentence_transformers import SentenceTransformer


def _to_np(x: Iterable) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x), dtype=np.float32)


def encode_documents(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    return _to_np(
        model.encode_document(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            convert_to_tensor=False,
            show_progress_bar=True,
        )
    )


def encode_query(
    text: str, model: SentenceTransformer, normalize: bool = True
) -> np.ndarray:
    return _to_np(
        model.encode_query(
            [text],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )
    )
