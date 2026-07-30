"""
Vision Service Module.
Handles image upload validation and delegates to the Grok service for analysis.
"""

import logging
from fastapi import UploadFile
from services.grok_service import analyze_image

logger = logging.getLogger(__name__)

# Allowed image MIME types
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Maximum file size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024


async def process_image(file: UploadFile) -> dict:
    """
    Validate and process an uploaded image file.

    Args:
        file: The uploaded image file from FastAPI.

    Returns:
        A dictionary containing the image analysis results from Grok.

    Raises:
        ValueError: If the file type is unsupported or the file is too large.
    """
    # Validate MIME type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type: '{content_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
        )

    # Read image bytes
    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large ({len(image_bytes) / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE / (1024 * 1024):.0f} MB."
        )

    if len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    logger.info(f"Processing image: {file.filename} ({content_type}, {len(image_bytes)} bytes)")

    # Delegate to Grok Vision API
    analysis = await analyze_image(image_bytes, content_type)

    return analysis
