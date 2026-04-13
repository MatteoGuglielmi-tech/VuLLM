import numpy as np
from sentence_transformers import SentenceTransformer


def encode(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 8,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vec = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        convert_to_tensor=False,
        show_progress_bar=True,
    )
    return np.asarray(vec)
