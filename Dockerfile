# Use PyTorch image with CUDA support
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda

# Set working directory
WORKDIR /app

# Install system dependencies including Ninja
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ninja-build \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Download pretrained models
RUN mkdir -p pretrained_models && \
    pip install gdown && \
    gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt && \
    wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat" -O pretrained_models/shape_predictor_68_face_landmarks.dat

# Copy the entire project
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]