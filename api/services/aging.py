import base64
from io import BytesIO
from typing import List, Optional
from fastapi import HTTPException
from api.schemas import aging as aging_schema
from loguru import logger
import numpy as np
from PIL import Image
import cv2
import dlib
import torch
from models.psp import pSp
from argparse import Namespace
import torchvision.transforms as transforms
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im

class AgingService:
    """Service class for face aging operations."""

    def __init__(self):
        """Initialize the AgingService with the face aging model and image transforms."""
        self.model = self._load_model()
        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _load_model(self) -> pSp:
        """
        Load the face aging model.

        Returns:
            pSp: The loaded face aging model.
        """
        logger.info("Loading face aging model...")
        model_path = "./pretrained_models/sam_ffhq_aging.pt"
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        logger.info("Face aging model loaded successfully")
        return net

    async def process_face_aging(
            self,
            base64_image: str,
            gender: aging_schema.Gender,
            cropping_model: aging_schema.CroppingModel,
            target_ages: Optional[List[int]] = None
    ) -> List[aging_schema.AgedImage]:
        """
        Process face aging on the input image.

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
            aged_images = self._generate_aged_images(face_image, target_ages or [10, 30, 50, 70])

            return [
                aging_schema.AgedImage(age=age, base64_image=img)
                for age, img in zip(target_ages or [10, 30, 50, 70], aged_images)
            ]
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in process_face_aging: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    def _base64_to_image(self, base64_image: str) -> Image.Image:
        """
        Convert a base64 encoded image to a PIL Image object.

        Args:
            base64_image (str): Base64 encoded image string.

        Returns:
            Image.Image: PIL Image object.

        Raises:
            HTTPException: If there's an error decoding the image.
        """
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decoding base64 image: {str(e)}")

    def _detect_and_crop_face(self, image: Image.Image, cropping_model: aging_schema.CroppingModel) -> Image.Image:
        """
        Detect and crop a face from the input image using the specified cropping model.

        Args:
            image (Image.Image): Input image.
            cropping_model (aging_schema.CroppingModel): Model to use for face detection and cropping.

        Returns:
            Image.Image: Cropped face image.
        """
        if cropping_model == aging_schema.CroppingModel.DLIB:
            return self._detect_and_crop_face_dlib(image)
        else:  # face_recognition
            return self._detect_and_crop_face_recognition(image)

    def _detect_and_crop_face_dlib(self, image: Image.Image) -> Image.Image:
        """
        Detect and crop a face from the input image using dlib.

        Args:
            image (Image.Image): Input image.

        Returns:
            Image.Image: Cropped face image.

        Raises:
            HTTPException: If no face or multiple faces are detected.
        """
        detector = dlib.get_frontal_face_detector()
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = detector(img_array, 1)

        if len(faces) == 0:
            raise HTTPException(status_code=422, detail="No face detected in the input image")
        elif len(faces) > 1:
            raise HTTPException(status_code=422, detail="Multiple faces detected. Please provide a single face photo.")

        face = faces[0]
        return self._crop_face_with_margin_dlib(image, face)

    def _crop_face_with_margin_dlib(self, image: Image.Image, face: dlib.rectangle) -> Image.Image:
        """
        Crop a face detected by dlib with additional margin.

        Args:
            image (Image.Image): Input image.
            face (dlib.rectangle): Detected face rectangle.

        Returns:
            Image.Image: Cropped face image with margin.
        """
        width, height = image.size
        center_x = (face.left() + face.right()) // 2
        center_y = (face.top() + face.bottom()) // 2

        face_width = int(face.width() * 2)
        face_height = int(face.height() * 2.5)

        left = max(0, center_x - face_width // 2)
        top = max(0, center_y - face_height // 2)
        right = min(width, center_x + face_width // 2)
        bottom = min(height, center_y + face_height // 2)

        top = max(0, top - int(face.height() * 0.5))

        return image.crop((left, top, right, bottom))

    def _detect_and_crop_face_recognition(self, image: Image.Image) -> Image.Image:
        """
        Detect and crop a face from the input image using face_recognition library.

        Args:
            image (Image.Image): Input image.

        Returns:
            Image.Image: Cropped face image.

        Note: This method is not implemented yet.
        """
        # Implement face_recognition-based cropping if needed
        pass

    def _generate_aged_images(self, face_image: Image.Image, target_ages: List[int]) -> List[str]:
        """
        Generate aged versions of the input face image for specified target ages.

        Args:
            face_image (Image.Image): Input face image.
            target_ages (List[int]): List of target ages for aging.

        Returns:
            List[str]: List of base64 encoded strings of the aged face images.
        """
        age_transformers = [AgeTransformer(target_age=age) for age in target_ages]
        images_list = []

        input_image = self.img_transforms(face_image)

        for age_transformer in age_transformers:
            with torch.no_grad():
                input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)
                result_tensor = self.model(input_image_age)[0]
                result_image = tensor2im(result_tensor)
                bs64_image = self._generate_base64_image(result_image)
                images_list.append(bs64_image)

        return images_list

    def _generate_base64_image(self, image: Image.Image) -> str:
        """
        Convert a PIL Image object to a base64 encoded string.

        Args:
            image (Image.Image): PIL Image object to be converted.

        Returns:
            str: Base64 encoded string of the image.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()