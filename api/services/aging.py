from typing import List, Optional, Tuple
from fastapi import HTTPException
from api.schemas import aging as aging_schema
from loguru import logger
import base64
from io import BytesIO
import face_recognition
from PIL import Image
import numpy as np
import dlib
import cv2
import replicate
import asyncio
import aiohttp


class AgingService:
    async def process_face_aging(
            self,
            base64_image: str,
            gender: aging_schema.Gender,
            cropping_model: aging_schema.CroppingModel,
            target_ages: Optional[List[int]] = None
    ) -> List[aging_schema.AgedImage]:
        """
        Process the input image and generate aged versions of the detected face.

        Args:
            base64_image (str): Base64 encoded input image.
            gender (aging_schema.Gender): Gender of the person in the image.
            cropping_model (aging_schema.CroppingModel): Model to use for face detection and cropping.
            target_ages (Optional[List[int]]): List of target ages for aging. Defaults to [10, 30, 50, 70].

        Returns:
            List[aging_schema.AgedImage]: List of aged images with their respective target ages.

        Raises:
            HTTPException: If an error occurs during processing.
        """
        try:
            image = self._base64_to_image(base64_image)
            face_image = self._detect_and_crop_face(image, cropping_model)
            original_face_base64 = self._generate_base64_image(face_image)

            aged_images = await self._generate_aged_images(original_face_base64, target_ages)

            return [
                aging_schema.AgedImage(age=age, base64_image=img)
                for age, img in zip(target_ages, aged_images)
            ]
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in process_face_aging: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    def _base64_to_image(self, base64_image: str) -> np.ndarray:
        """
        Convert a base64 encoded image to a numpy array.

        Args:
            base64_image (str): Base64 encoded image string.

        Returns:
            np.ndarray: Image as a numpy array.

        Raises:
            HTTPException: If there's an error decoding the image.
        """
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            return np.array(image, dtype=np.uint8)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decoding base64 image: {str(e)}")

    def _detect_and_crop_face(self, image: np.ndarray, cropping_model: aging_schema.CroppingModel) -> np.ndarray:
        """
        Detect and crop a face from the input image using the specified cropping model.

        Args:
            image (np.ndarray): Input image as a numpy array.
            cropping_model (aging_schema.CroppingModel): Model to use for face detection and cropping.

        Returns:
            np.ndarray: Cropped face image.
        """
        if cropping_model == aging_schema.CroppingModel.DLIB:
            return self._detect_and_crop_face_dlib(image)
        else:  # face_recognition
            return self._detect_and_crop_face_recognition(image)

    def _detect_and_crop_face_dlib(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and crop a face from the input image using dlib.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Cropped face image.

        Raises:
            HTTPException: If no face or multiple faces are detected.
        """
        detector = dlib.get_frontal_face_detector()
        img_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = detector(img_array, 1)

        if len(faces) == 0:
            raise HTTPException(status_code=422, detail="No face detected in the input image")
        elif len(faces) > 1:
            raise HTTPException(status_code=422, detail="Multiple faces detected. Please provide a single face photo.")

        face = faces[0]
        return self._crop_face_with_margin_dlib(image, face)

    def _crop_face_with_margin_dlib(self, image: np.ndarray, face: dlib.rectangle) -> np.ndarray:
        """
        Crop a face detected by dlib with additional margin.

        Args:
            image (np.ndarray): Input image as a numpy array.
            face (dlib.rectangle): Detected face rectangle.

        Returns:
            np.ndarray: Cropped face image with margin.
        """
        height, width = image.shape[:2]
        center_x = (face.left() + face.right()) // 2
        center_y = (face.top() + face.bottom()) // 2

        face_width = int(face.width() * 2)
        face_height = int(face.height() * 2.5)

        left = max(0, center_x - face_width // 2)
        top = max(0, center_y - face_height // 2)
        right = min(width, center_x + face_width // 2)
        bottom = min(height, center_y + face_height // 2)

        top = max(0, top - int(face.height() * 0.5))

        return image[top:bottom, left:right]

    def _detect_and_crop_face_recognition(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and crop a face from the input image using face_recognition library.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Cropped face image.

        Raises:
            HTTPException: If no face or multiple faces are detected.
        """
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            raise HTTPException(status_code=422, detail="No face detected in the input image")
        elif len(face_locations) > 1:
            raise HTTPException(status_code=422, detail="Multiple faces detected. Please provide a single face photo.")

        return self._crop_face_with_margin_recognition(image, face_locations[0])

    def _crop_face_with_margin_recognition(self, image: np.ndarray,
                                           face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop a face detected by face_recognition with additional margin.

        Args:
            image (np.ndarray): Input image as a numpy array.
            face_location (Tuple[int, int, int, int]): Detected face location.

        Returns:
            np.ndarray: Cropped face image with margin.
        """
        top, right, bottom, left = face_location
        height, width = image.shape[:2]

        face_height = bottom - top
        face_width = right - left

        margin_v = int(face_height * 0.5)
        margin_h = int(face_width * 0.5)

        new_top = max(0, top - margin_v)
        new_bottom = min(height, bottom + margin_v)
        new_left = max(0, left - margin_h)
        new_right = min(width, right + margin_h)

        return image[new_top:new_bottom, new_left:new_right]

    def _generate_base64_image(self, face_image: np.ndarray) -> str:
        """
        Convert a numpy array image to a base64 encoded string.

        Args:
            face_image (np.ndarray): Input face image as a numpy array.

        Returns:
            str: Base64 encoded image string.
        """
        pil_image = Image.fromarray(face_image)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")

        return base64.b64encode(buffered.getvalue()).decode()

    async def _generate_aged_images(self, base64_image: str, target_ages: List[int]) -> List[str]:
        """
        Generate aged versions of the input image for the specified target ages.

        Args:
            base64_image (str): Base64 encoded input image.
            target_ages (List[int]): List of target ages for aging.

        Returns:
            List[str]: List of base64 encoded aged images.

        Raises:
            HTTPException: If an error occurs during the aging process.
        """

        async def process_single_age(age: str) -> str:
            try:
                input_data = {
                    "image": f"data:image/png;base64,{base64_image}",
                    "target_age": age
                }
                prediction = await replicate.predictions.async_create(
                    version="9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
                    input=input_data
                )

                while prediction.status != "succeeded":
                    await asyncio.sleep(1)
                    prediction = await replicate.predictions.async_get(prediction.id)

                if prediction.status == "failed":
                    raise HTTPException(status_code=500,
                                        detail=f"Face aging API failed to process the image for age {age}")

                if isinstance(prediction.output, str):
                    return prediction.output
                elif isinstance(prediction.output, list):
                    for item in prediction.output:
                        if isinstance(item, dict) and 'file' in item:
                            return item['file']
                        elif isinstance(item, str):
                            return item

                raise HTTPException(status_code=500,
                                    detail=f"Unexpected output format from Face aging API for age {age}: {prediction.output}")

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error in face aging API for age {age}: {str(e)}")

        try:
            aged_image_urls = await asyncio.gather(*[process_single_age(str(age)) for age in target_ages])

            aged_image_urls = [url for url in aged_image_urls if url is not None]

            if not aged_image_urls:
                raise HTTPException(status_code=500, detail="No valid image URLs found in any of the API responses")

            aged_images_base64 = await asyncio.gather(*[self._download_and_encode(url) for url in aged_image_urls])
            return aged_images_base64

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in face aging API: {str(e)}")

    async def _download_and_encode(self, url: str) -> str:
        """
        Download an image from a URL and encode it as base64.

        Args:
            url (str): URL of the image to download.

        Returns:
            str: Base64 encoded image string.

        Raises:
            HTTPException: If the image download fails.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return base64.b64encode(image_data).decode('utf-8')
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to download image from {url}")