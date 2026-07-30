"""
Chatbot Routes Module.
Defines the API endpoint for scientific figure analysis and paper recommendation.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from services.vision_service import process_image
from services.recommendation_service import get_recommendations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Recommendation"])


@router.post("/recommend")
async def recommend_papers(image: UploadFile = File(...)):
    """
    Accept a scientific figure image and return recommended research papers.

    - Analyzes the image using Grok Vision API
    - Performs semantic search against the paper dataset
    - Returns top-5 papers with LLM-generated relevance explanations

    Args:
        image: The uploaded scientific figure (multipart/form-data).

    Returns:
        JSON with image_description, keywords, domain, and recommended_papers.
    """
    try:
        # Step 1: Validate and analyze the image
        logger.info(f"Received image: {image.filename}")
        image_analysis = await process_image(image)

        # Step 2-6: Get full recommendations (embed → search → explain)
        response = await get_recommendations(image_analysis)

        return response

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error in /api/recommend: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request."
        )
