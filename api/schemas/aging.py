import enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, conlist

class Gender(enum.Enum):
    """
    Enum representing the gender of the person in the image.
    """
    MAN = "MAN"
    WOMAN = "WOMAN"

class CroppingModel(enum.Enum):
    """
    Enum representing the model to use for face detection and cropping.
    """
    DLIB = "DLIB"
    FACE_RECOGNITION = "FACE_RECOGNITION"

class AgeDetectionInput(BaseModel):
    """
    Input model for the age detection and face aging process.

    Attributes:
        base64_image: A base64 encoded string of the image containing a face.
        gender: The gender of the person in the image.
        cropping_model: The model to use for face detection and cropping.
        target_ages: Optional list of target ages for aging. Each age should be between 0 and 100.
    """
    base64_image: str = Field(
        ..., description="Base64 encoded image containing a face"
    )
    gender: Gender = Field(
        ..., description="Gender of the person in the image (MAN or WOMAN)"
    )
    cropping_model: Optional[CroppingModel] = Field(
        default=CroppingModel.DLIB,
        description="Model to use for face detection and cropping"
    )
    target_ages: Optional[List[int]] = Field(
        default=[10, 30, 50, 70],
        description="List of target ages for aging. Each age should be between 0 and 100. Max 10 ages."
    )

    @field_validator('target_ages')
    def validate_target_ages(cls, target_ages):
        """
        Validate that all target ages are between 0 and 100.

        Args:
            target_ages (List[int]): List of target ages to validate.

        Raises:
            ValueError: If any age is not between 0 and 100.

        Returns:
            List[int]: The validated list of target ages.
        """
        if not all(0 <= age <= 100 for age in target_ages):
            raise ValueError("All target ages must be between 0 and 100")
        return target_ages

class AgedImage(BaseModel):
    """
    Model representing a single aged image.

    Attributes:
        age: The target age for this aged image.
        base64_image: A base64 encoded string of the aged image.
    """
    age: int = Field(..., description="The target age for this aged image")
    base64_image: str = Field(..., description="Base64 encoded aged image")

class AgingResponse(BaseModel):
    """
    Response model for the face aging process.

    Attributes:
        aged_images: A list of AgedImage objects representing the aged versions of the input face.
    """
    aged_images: List[AgedImage] = Field(
        ..., description="List of aged images with their respective target ages"
    )