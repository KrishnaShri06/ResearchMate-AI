"""
Recommendation Service Module.
Orchestrates the full recommendation pipeline: semantic search + LLM explanations.
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from services.embedding_service import get_query_embedding, get_paper_embeddings
from services.grok_service import explain_recommendations
from utils.config import TOP_K

logger = logging.getLogger(__name__)


def find_similar_papers(query_embedding: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    """
    Find the most similar papers using cosine similarity.

    Args:
        query_embedding: The embedding vector of the query text.
        top_k: Number of top results to return.

    Returns:
        A list of dicts with 'title', 'abstract', and 'similarity' keys,
        sorted by descending similarity.
    """
    paper_embeddings, papers_data = get_paper_embeddings()

    # Reshape query for cosine_similarity: (1, dim) vs (n, dim)
    query_reshaped = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_reshaped, paper_embeddings)[0]

    # Get top-k indices sorted by similarity (descending)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "title": papers_data[idx]["title"],
            "abstract": papers_data[idx]["abstract"],
            "similarity": round(float(similarities[idx]), 4),
        })

    logger.info(f"Found top {len(results)} similar papers. Best score: {results[0]['similarity']:.4f}")
    return results


async def get_recommendations(image_analysis: dict) -> dict:
    """
    Full recommendation pipeline.

    1. Generate query embedding from image description
    2. Find top-k similar papers via cosine similarity
    3. Get LLM explanations for each recommendation
    4. Assemble the final response

    Args:
        image_analysis: Dict from Grok Vision containing 'description', 'keywords', etc.

    Returns:
        Final response dict with 'image_description', 'keywords', and 'recommended_papers'.
    """
    description = image_analysis.get("description", "")
    keywords = image_analysis.get("keywords", [])

    # Build a rich query from multiple analysis fields for better retrieval
    query_parts = [description]
    if keywords:
        query_parts.append("Keywords: " + ", ".join(keywords))
    if image_analysis.get("domain"):
        query_parts.append(f"Domain: {image_analysis['domain']}")
    if image_analysis.get("methodology"):
        query_parts.append(f"Methodology: {image_analysis['methodology']}")

    query_text = " ".join(query_parts)

    # Step 1: Generate query embedding
    logger.info("Generating query embedding...")
    query_embedding = get_query_embedding(query_text)

    # Step 2: Semantic search
    logger.info("Performing semantic search...")
    similar_papers = find_similar_papers(query_embedding)

    # Step 3: Get LLM explanations
    logger.info("Generating LLM explanations for recommendations...")
    explanations = await explain_recommendations(description, keywords, similar_papers)

    # Step 4: Merge explanations into paper results
    explanation_map = {item["title"]: item["reason"] for item in explanations}

    recommended_papers = []
    for paper in similar_papers:
        recommended_papers.append({
            "title": paper["title"],
            "abstract": paper["abstract"],
            "similarity": paper["similarity"],
            "reason": explanation_map.get(paper["title"], "Relevant based on semantic similarity."),
        })

    # Build final response
    response = {
        "image_description": description,
        "keywords": keywords,
        "domain": image_analysis.get("domain", "Unknown"),
        "recommended_papers": recommended_papers,
    }

    logger.info(f"Recommendation pipeline complete. Returning {len(recommended_papers)} papers.")
    return response
