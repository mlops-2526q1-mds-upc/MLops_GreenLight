# Base image with CUDA 12.1
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Working directory
WORKDIR /workspace

# Avoid Python output buffering
ENV PYTHONUNBUFFERED=1

# ------------------------------
# Install system dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    cmake \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN python3 -m pip install --upgrade pip

# ------------------------------
# Install latest compatible PyTorch, torchvision, torchaudio
# ------------------------------
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ------------------------------
# Install Detectron2 from source
# ------------------------------
RUN git clone https://github.com/facebookresearch/detectron2.git /workspace/detectron2 && \
    cd /workspace/detectron2 && \
    python3 -m pip install .

# ------------------------------
# Install additional ML libraries
# ------------------------------
RUN pip install opencv-python matplotlib tqdm jupyter notebook pyyaml omegaconf hydra-core codecarbon mlflow uvc dagshub dvc

# ------------------------------
# Expose Jupyter port
# ------------------------------
EXPOSE 8888

# Default command
CMD ["bash"]

