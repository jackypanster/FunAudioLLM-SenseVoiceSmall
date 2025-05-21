# FunAudioLLM - SenseVoiceSmall ASR API

This project provides a RESTful API服务 for speech-to-text (ASR) transcription using the FunAudioLLM/SenseVoiceSmall model. It is built with FastAPI.

**Current Status: Initial setup complete, requires actual model loading and inference logic implementation.**

## Features

*   Exposes an `/asr_pure` endpoint for transcribing audio files.
*   Exposes a `/health` endpoint for service health checks.
*   Uses `uv` for Python environment and package management.
*   Designed for deployment on NVIDIA GPU enabled servers.
*   Basic multi-GPU considerations (selectable via environment variables).

## Project Structure

```
FunAudioLLM-SenseVoiceSmall/
├── app/                      # FastAPI application code
│   ├── __init__.py
│   ├── main.py               # FastAPI app, routes, lifespan management
│   ├── models/               # Model loading logic
│   │   ├── __init__.py
│   │   └── sensevoice_loader.py # Loads SenseVoiceSmall model
│   ├── services/             # Core ASR service logic
│   │   ├── __init__.py
│   │   └── asr_service.py    # Handles audio processing and transcription
│   └── schemas.py            # Pydantic models for API I/O
├── .gitignore                # Standard Python gitignore
├── architecture.md           # System architecture design
├── detailed_design.md        # Detailed component design
├── deployment_guide.md       # Step-by-step deployment instructions
├── pro.md                    # Initial project requirements
├── README.md                 # This file
├── requirements.txt          # Python package dependencies
└── test_cases.md             # Test cases for the service
```

## Prerequisites

*   Python 3.8+ (uv will manage the virtual environment with a compatible version)
*   `uv` installed ([https://docs.astral.sh/uv/guides/install/](https://docs.astral.sh/uv/guides/install/))
*   NVIDIA GPU with appropriate drivers and CUDA installed (see `pro.md` for tested environment).
*   SenseVoiceSmall model files downloaded to a known location (default: `/home/llm/model/iic/SenseVoiceSmall`).

## Setup and Installation (Remote Server)

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <your-repo-url> FunAudioLLM-SenseVoiceSmall
    cd FunAudioLLM-SenseVoiceSmall
    ```

2.  **Create and activate virtual environment using uv:**
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
    *Ensure `numpy` is included and installed, as it's a critical dependency for PyTorch/Torchaudio.*

4.  **Configure Model Path & Device (Environment Variables - Optional):**
    The application can be configured using the following environment variables:
    *   `MODEL_PATH`: Path to the SenseVoiceSmall model directory (default: `/home/llm/model/iic/SenseVoiceSmall`).
    *   `DEVICE`: PyTorch device string (e.g., `cuda:0`, `cuda:1`, `cpu`; default: `cuda:0` if available, else `cpu`).
    *   `LOG_LEVEL`: Logging level (e.g., `INFO`, `DEBUG`; default: `INFO`).

    Set these before running the application if you need to override defaults.
    Example:
    ```bash
    export MODEL_PATH="/path/to/your/SenseVoiceSmall"
    export DEVICE="cuda:1"
    ```

5.  **Implement Model Logic:**
    *   **CRITICAL:** Modify `app/models/sensevoice_loader.py` to replace the placeholder `"dummy_model_object"` with the actual code to load your SenseVoiceSmall model.
    *   **CRITICAL:** Modify `app/services/asr_service.py` to replace the placeholder transcription logic with actual model inference calls using the loaded model.

## Running the Application (Remote Server)

Once setup and model logic are implemented:

```bash
# Ensure .venv is activated: source .venv/bin/activate
uv run uvicorn app.main:app --host 0.0.0.0 --port 8888 --workers 1
```

*   `--host 0.0.0.0`: Makes the server accessible externally.
*   `--port 8000`: Port to run on. (You used 8888 in your test, adjust as needed).
*   `--workers 1`: Number of Uvicorn workers. For GPU-bound tasks, typically 1 worker per GPU is a starting point.

To use a specific GPU (e.g., the second GPU, which is index 1):
```bash
export DEVICE="cuda:1" # or CUDA_VISIBLE_DEVICES=1 and DEVICE="cuda:0"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8888 --workers 1
```

Refer to `deployment_guide.md` for more advanced deployment options, including using `systemd` or `supervisor` for service management and multi-GPU setups.

## API Endpoints

*   **`POST /asr_pure`**: 
    *   Upload an audio file (e.g., `.wav`, `.mp3`) for transcription.
    *   Request: `multipart/form-data` with a `file` field.
    *   Successful Response (200 OK):
        ```json
        {
          "text": "Transcribed text...",
          "status": "success",
          "processing_time_ms": 123.45
        }
        ```
*   **`GET /health`**:
    *   Returns the health status of the service, indicating if the model is loaded.
    *   Successful Response (200 OK or 503 if unhealthy):
        ```json
        // Healthy
        {"status": "ok", "message": "ASR service is healthy and model is loaded."}
        // Degraded (dummy model)
        {"status": "degraded", "message": "ASR service is running with a DUMMY model. Transcription will not work."}
        // Unhealthy
        {"status": "unhealthy", "message": "ASR model loader not available or model failed to load."}
        ```

## Development Notes

*   The dummy model placeholders in `sensevoice_loader.py` and `asr_service.py` **must** be replaced with actual model loading and inference code for the service to be functional.
*   Ensure all dependencies, especially `torch`, `torchaudio`, and `numpy`, are correctly installed in the `uv` environment on the target machine.
*   Consult `test_cases.md` for examples on how to interact with the API.

## Next Steps (Critical)

1.  **Implement Model Loading**: Update `app/models/sensevoice_loader.py`.
2.  **Implement Inference Logic**: Update `app/services/asr_service.py`.
3.  **Thoroughly Test**: Use the test audio samples and cases from `test_cases.md`. 