# Use NVIDIA CUDA base image for GPU support with CUDA 12.9
# PyTorch 2.3.1 is built against CUDA 12.1 but is forward compatible with CUDA 12.9
FROM nvidia/cuda:12.9.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PYSETUP_PATH="/opt/pysetup" \
    WORKDIR="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    python3.10 \
    python3-pip \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 
ENV PATH="${PATH}:${POETRY_HOME}/bin"

# Set working directory
WORKDIR ${WORKDIR}

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.9 runtime)
RUN pip3 install --no-cache-dir torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-dev --no-root && \
    rm -rf "$POETRY_CACHE_DIR"

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/static/uploads /app/diarization_output /app/transcript_output

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
