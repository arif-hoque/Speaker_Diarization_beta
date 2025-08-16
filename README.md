# Audio Diarization API

A FastAPI-based service for audio diarization using Whisper and Pyannote models. This service provides an API endpoint to upload audio files and receive speaker-separated transcripts with timestamps.

## Features

- Audio file upload and processing
- Speaker diarization using Pyannote
- Speech-to-text transcription using Whisper
- Support for multiple audio formats (WAV, MP3, OGG, FLAC, M4A)
- GPU acceleration support
- Containerized with Docker for easy deployment

## Prerequisites

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- Hugging Face authentication token (for Pyannote model)

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd fastapi-diarization
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```bash
# Hugging Face token (required for Pyannote models)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Optional: Set to 'true' for debug mode
DEBUG=false

# Optional: Set host and port
HOST=0.0.0.0
PORT=5000
```

### 3. Build and run with Docker (Recommended)

```bash
# Build the Docker image
docker build -t fastapi-diarization .

# Run the container with GPU support
docker run --gpus all -p 5000:5000 \
  -v $(pwd)/diarization_output:/app/diarization_output \
  -v $(pwd)/transcript_output:/app/transcript_output \
  -v $(pwd)/static/uploads:/app/static/uploads \
  --env-file .env \
  fastapi-diarization


# Run the container with GPU support and hot reload
docker run --gpus all -p 5000:5000 \
  -v $(pwd)/diarization_output:/app/diarization_output \
  -v $(pwd)/transcript_output:/app/transcript_output \
  -v $(pwd)/static/uploads:/app/static/uploads \
  -v $(pwd)/app:/app/app \
  --env-file .env \
  fastapi-diarization \
  uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

### 4. Or run locally with Poetry

```bash
# Install dependencies
poetry install

# Run the application
poetry run uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

## API Documentation

Once the service is running, you can access:

- **API Documentation**: http://localhost:5000/docs
- **Redoc Documentation**: http://localhost:5000/redoc

## API Endpoints

### Process Audio

```
POST /api/diarization/process
```

**Request:**
- `file`: Audio file to process (required)
- `num_speakers`: Number of speakers (optional, auto-detect if not provided)
- `language`: Language code (e.g., 'en', 'es', 'fr') - optional, auto-detect if not provided
- `translate`: Whether to translate to English (default: false)
- `prompt`: Optional prompt for better transcription (e.g., names, technical terms)

**Response:**
```json
{
  "language": "en",
  "num_speakers": 2,
  "duration": 120.5,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, this is the first speaker.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99},
        {"word": ",", "start": 0.5, "end": 0.6, "probability": 0.98},
        ...
      ]
    },
    ...
  ]
}
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
# Format code with Black and isort
poetry run black .
poetry run isort .
```

## Deployment

### Kubernetes

For production deployment, you can use the provided Kubernetes manifests in the `k8s/` directory.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | Yes | - | Hugging Face authentication token |
| `DEBUG` | No | `false` | Enable debug mode |
| `HOST` | No | `0.0.0.0` | Host to bind the server to |
| `PORT` | No | `5000` | Port to run the server on |

