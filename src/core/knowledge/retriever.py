import logging
from functools import partial
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

from .cwe_knowledge_base import CWEEntry, CWEKnowledgeBase
from .embedding_cache import EncodeFn, FloatMatrix, IntVector, load_or_build
from .encoder import encode_documents, encode_query

logger = logging.getLogger(name=__name__)


def should_index(entry: CWEEntry) -> bool:
    platforms: set[str] = {p.name for p in entry.platforms}
    if platforms & {"C", "C++", "Not Language-Specific"} or entry.abstraction in {
        "Pillar",
        "Class",
    }:
        return True

    return False


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CWERetriever:
    def __init__(
        self,
        kb: CWEKnowledgeBase,
        embedding_model_name: str,
        caching_dir: Path,
        batch_size: int = 32,
    ) -> None:
        self._kb = kb
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        docs = self._get_docs_to_index()
        encode_fn: EncodeFn = self._build_encodefn(batch_size)

        vectors, ids = self._get_embeds(
            docs2index=docs,
            embedding_model_name=embedding_model_name,
            caching_dir=caching_dir,
            encode_fn=encode_fn,
        )

        # vectors: (N, D) float32, normalized (L2 norm = 1) for cosine
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)  # type: ignore[reportCallIssue]
        self._row_to_cwe_id = ids

        logger.info(
            f"CWERetriever: indexed {self._index.ntotal} CWEs, dim={vectors.shape[1]}"
        )

    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        device: str = pick_device()
        lmodel = SentenceTransformer(model_name, device=device)
        logger.info(f"{model_name} loaded on device={device}")
        return lmodel

    def _get_docs_to_index(self) -> list[CWEEntry]:
        docs = [e for e in self._kb.get_all() if should_index(e)]
        logger.info(
            f"Indexing {len(docs)}/{len(self._kb)} CWEs (filtered by C/C++/NLS + Pillar/Class)"
        )

        return docs

    def _build_encodefn(self, batch_size: int) -> EncodeFn:
        return partial(
            encode_documents,
            model=self.embedding_model,
            normalize=True,  # required for IndexFlatIP cosine
            batch_size=batch_size,
        )

    def _get_embeds(
        self,
        docs2index: list[CWEEntry],
        embedding_model_name: str,
        caching_dir: Path,
        encode_fn: EncodeFn,
    ) -> tuple[FloatMatrix, IntVector]:
        try:
            vectors, ids = load_or_build(
                docs=docs2index,
                model_name=embedding_model_name,
                cache_dir=caching_dir,
                encode_fn=encode_fn,
            )
            logger.info(f"CWERetriever: {vectors.shape[0]} CWEs encoded")
        except Exception:
            logger.exception("Failed to build/load embeddings")
            raise
        else:
            return vectors, ids

    def retrieve(self, query: str, k: int) -> list[tuple[CWEEntry, float]]:
        q_vec = encode_query(text=query, model=self.embedding_model)
        scores, rows = self._index.search(q_vec, k)  # type: ignore[reportCallIssue]
        logger.debug(
            f"CWERetriever: query len={len(query)}, top_score={scores[0][0]:.3f}"
        )

        top = float(scores[0][0])
        if top < 0.4:
            logger.warning(
                f"Weak retrieval: top_score={top:.3f} (query len={len(query)})"
            )
        else:
            logger.debug(f"retrieve: top_score={top:.3f}, k={k}")

        # rows: (1, k) array of FAISS row indices
        cwe_ids = self._row_to_cwe_id[rows[0]]  # (k,)
        entries = [self._kb.require(int(c)) for c in cwe_ids]

        return list(zip(entries, scores[0].tolist()))
