# Face Aging API

## Project Overview

The Face Aging API is a sophisticated service that leverages deep learning techniques to generate aged versions of facial images. This project utilizes FastAPI to provide a robust and efficient API interface for face aging operations.

### Key Features:
- Face detection and cropping using dlib or face_recognition
- Image aging for multiple target ages using SAM (Style-based Age Manipulation)
- Base64 image encoding/decoding for seamless data transfer
- In-app processing without relying on external APIs

### Technologies Used:
- FastAPI for API development
- dlib and face_recognition for face detection and cropping
- PyTorch and SAM model for face aging
- Docker for containerization and deployment

## Project Structure

```
face-aging-api/
│
├── api/
│   ├── routers/
│   │   └── aging.py
│   ├── schemas/
│   │   └── aging.py
│   ├── services/
│   │   └── aging.py
│   └── __init__.py
│
├── configs/       # SAM model configuration files
├── datasets/      # SAM model dataset utilities
├── models/        # SAM model architecture
├── utils/         # SAM model utility functions
│
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Key Differences from Main Version

1. **In-App Processing**: The new version performs face aging within the application, eliminating the need for external API calls.

2. **SAM Model Integration**: I've integrated the SAM (Style-based Age Manipulation) model directly into our project. The `configs`, `datasets`, `models`, and `utils` directories contain the necessary files for the SAM model.

4. **Optimized Performance**: The face aging model is loaded once at startup and reused for all requests, improving response times.

5. **Docker Optimization**: The Dockerfile has been updated to include all necessary dependencies and model files for standalone operation.

## Installation Guide

### Docker Installation (Recommended)

1. Build the Docker image:
   ```
   docker build -t face-aging-api .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 face-aging-api
   ```


## API Usage

### Face Aging Endpoint

- **URL**: `/face-aging`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "base64_image": "base64_encoded_image_string",
    "gender": "MAN" or "WOMAN",
    "cropping_model": "DLIB" or "FACE_RECOGNITION",
    "target_ages": [10, 30, 50, 70]
  }
  ```

- **Response**:
  ```json
  {
    "aged_images": [
      {
        "age": 10,
        "base64_image": "base64_encoded_image_string_age_10"
      },
      {
        "age": 30,
        "base64_image": "base64_encoded_image_string_age_30"
      },
      {
        "age": 50,
        "base64_image": "base64_encoded_image_string_age_50"
      },
      {
        "age": 70,
        "base64_image": "base64_encoded_image_string_age_70"
      }
    ]
  }
  ```

## API Documentation

FastAPI automatically generates interactive API documentation. After starting the server, visit:

- Swagger UI: `http://localhost:8000/docs`
