"""
Embedding Service Module.
Handles sentence embedding generation and caching for research papers and queries.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Module-level globals for cached state ────────────────────────────
_model: SentenceTransformer = None
_paper_embeddings: np.ndarray = None
_papers_data: list[dict] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _model


def init_embeddings(papers_df) -> None:
    """
    Generate and cache embeddings for all paper abstracts.
    Called once at server startup.

    Args:
        papers_df: Pandas DataFrame with 'title' and 'abstract' columns.
    """
    global _paper_embeddings, _papers_data

    model = _get_model()

    abstracts = papers_df["abstract"].tolist()
    titles = papers_df["title"].tolist()

    logger.info(f"Generating embeddings for {len(abstracts)} paper abstracts...")
    _paper_embeddings = model.encode(abstracts, show_progress_bar=True, convert_to_numpy=True)

    # Store paper metadata alongside embeddings
    _papers_data = []
    for i, row in papers_df.iterrows():
        paper_dict = {"title": titles[i], "abstract": abstracts[i]}
        # Include optional columns if present
        for col in ["authors", "categories", "year"]:
            if col in papers_df.columns:
                paper_dict[col] = row[col]
        _papers_data.append(paper_dict)

    logger.info(f"Embeddings cached: {_paper_embeddings.shape}")


def get_query_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for a single query text.

    Args:
        text: The text to embed (e.g., image description).

    Returns:
        A numpy array representing the text embedding.
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def get_paper_embeddings() -> tuple[np.ndarray, list[dict]]:
    """
    Retrieve the cached paper embeddings and metadata.

    Returns:
        A tuple of (embeddings_array, papers_data_list).

    Raises:
        RuntimeError: If embeddings have not been initialized yet.
    """
    if _paper_embeddings is None or _papers_data is None:
        raise RuntimeError(
            "Paper embeddings have not been initialized. "
            "Ensure init_embeddings() is called at server startup."
        )
    return _paper_embeddings, _papers_data
