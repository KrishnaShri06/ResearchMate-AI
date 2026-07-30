"""
Configuration module for the Research Assistant backend.
Loads environment variables and exposes application constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Groq API Configuration ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"
GROQ_TEXT_MODEL = "llama-3.3-70b-specdec"

# ── Embedding Configuration ─────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Dataset Configuration ───────────────────────────────────────────
PROCESSED_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_processed_papers.csv")
HF_DATASET_NAME = "CCRss/arXiv_dataset"
HF_DATASET_SPLIT = "train[:5000]"

# ── Recommendation Settings ─────────────────────────────────────────
TOP_K = 5
