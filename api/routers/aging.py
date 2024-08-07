from fastapi import APIRouter, HTTPException
from api.schemas import aging as aging_schema
from api.services import aging as aging_service
from typing import List
from loguru import logger

router = APIRouter()


@router.post("/face-aging", response_model=List[aging_schema.AgedImage],status_code=200)
async def face_aging(
        input_data: aging_schema.AgeDetectionInput
) -> List[aging_schema.AgedImage]:
    """
    Perform face aging on the input image.

    This endpoint takes a base64 encoded image and generates aged versions of the face detected in the image.
    It returns a list of aged images with their respective target ages.

    Args:
        input_data (aging_schema.AgeDetectionInput): The input data containing the base64 image,
                                                     gender, cropping model, and target ages.

    Returns:
        List[aging_schema.AgedImage]: A list of aged images with their respective target ages.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        logger.info("Received face aging request")

        aging_service_instance = aging_service.AgingService()
        aged_images = await aging_service_instance.process_face_aging(
            input_data.base64_image,
            input_data.gender,
            input_data.cropping_model,
            input_data.target_ages
        )

        logger.info("Face aging completed successfully")
        return aged_images

    except HTTPException as e:
        logger.error(f"HTTP exception in aging service: {e.detail}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")