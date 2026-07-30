"""
Paper Loader Module.
Handles downloading the Hugging Face arXiv dataset, filtering it,
caching it locally to a CSV, and loading it on subsequent starts.
"""

import os
import logging
import pandas as pd
from datasets import load_dataset
from utils.config import HF_DATASET_NAME, HF_DATASET_SPLIT

logger = logging.getLogger(__name__)

# Target categories to filter for AI/ML papers
AI_CATEGORIES = {"cs.AI", "cs.CV", "cs.LG", "cs.CL"}

def _matches_ai(cats):
    """Check if the paper belongs to AI-related categories."""
    if pd.isna(cats):
        return False
    
    # arXiv categories are often strings with space-separated values, or arrays.
    if isinstance(cats, list):
        cat_list = cats
    else:
        # e.g., "cs.CV cs.LG" inside a string
        cat_list = [c.strip() for c in str(cats).replace(",", " ").split()]
        
    return any(c in AI_CATEGORIES for c in cat_list)

def _load_from_csv(csv_path: str) -> pd.DataFrame:
    """Helper to load and validate from cached CSV."""
    df = pd.read_csv(csv_path)
    # Ensure required columns
    required = {"title", "abstract"}
    if not required.issubset(set(df.columns.str.lower())):
        raise ValueError("Cached CSV is missing required columns.")
    
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna(subset=["title", "abstract"])
    df = df.reset_index(drop=True)
    return df

def load_papers(cache_path: str) -> pd.DataFrame:
    """
    Load research papers from a local CSV cache if it exists.
    Otherwise, download from Hugging Face, filter, cache, and return.

    Args:
        cache_path: Path to the local CSV cache.

    Returns:
        A cleaned pandas DataFrame with 'title', 'abstract', 'authors', 'categories'.
    """
    # 1. Check local cache first
    if os.path.exists(cache_path):
        try:
            logger.info(f"Local dataset cache found at {cache_path}. Loading...")
            df = _load_from_csv(cache_path)
            logger.info(f"Loaded {len(df)} papers from cache.")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Will attempt to download from Hugging Face.")

    # 2. Download from Hugging Face
    logger.info(f"Downloading dataset {HF_DATASET_NAME} from Hugging Face (streaming mode)...")
    try:
        dataset = load_dataset(HF_DATASET_NAME, split="train", streaming=True)
        # We manually take up to 5000 samples to prevent massive downloads.
        # This keeps memory overhead strictly small as requested.
        dataset_head = dataset.take(5000)
        df = pd.DataFrame(list(dataset_head))
    except Exception as e:
        logger.error(f"Failed to download Hugging Face dataset: {e}")
        # Try a last-ditch fallback to the sample 'research_papers.csv' if it exists in the same dir
        fallback_path = os.path.join(os.path.dirname(cache_path), "research_papers.csv")
        if os.path.exists(fallback_path):
            logger.warning(f"Falling back to original sample dataset: {fallback_path}")
            return _load_from_csv(fallback_path)
        raise e

    logger.info(f"Downloaded {len(df)} raw papers. Processing and filtering...")

    # Normalize column names to lowercase just in case
    df.columns = df.columns.str.lower().str.strip()

    # Ensure required columns exist
    for col in ["title", "abstract", "authors", "categories"]:
        if col not in df.columns:
            df[col] = "Unknown"  # Stub missing non-critical columns

    # 3. Process and filter
    df = df[["title", "abstract", "authors", "categories"]]
    initial_len = len(df)
    
    # Drop empty title/abstract
    df = df.dropna(subset=["title", "abstract"])
    
    # Filter for AI categories
    df = df[df["categories"].apply(_matches_ai)]
    
    df = df.reset_index(drop=True)
    logger.info(f"Filtered out {initial_len - len(df)} non-AI or invalid papers. Retained {len(df)} papers.")

    if len(df) == 0:
        logger.warning("No papers matched the AI filters in the downloaded batch!")

    # 4. Save to local cache
    logger.info(f"Saving processed dataset to cache: {cache_path}")
    df.to_csv(cache_path, index=False)

    return df
