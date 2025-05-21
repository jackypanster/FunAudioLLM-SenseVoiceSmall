import logging
import os
import asyncio # Make sure asyncio is imported, especially for asr_service placeholders
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware # Optional: if you need CORS

from app.schemas import ASRResponse, ErrorResponse
from app.models.sensevoice_loader import SenseVoiceLoader, ModelLoadError
from app.services.asr_service import ASRService, AudioProcessingError, ModelInferenceError

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() # Outputs to console
        # You can add logging.FileHandler here to log to a file, e.g.:
        # logging.FileHandler("asr_service.log")
    ]
)
logger = logging.getLogger(__name__)

# --- Global Variables / State ---
# (Populated during lifespan events)
app_state = {}

@asynccontextmanager
async def lifespan(app_instance: FastAPI): # Renamed app to app_instance to avoid conflict with global 'app'
    """
    Context manager to handle startup and shutdown events.
    Loads the model on startup and cleans up on shutdown (if necessary).
    """
    logger.info("Application startup: Initializing model loader...")
    try:
        model_loader = SenseVoiceLoader() # Instantiates the singleton
        # Trigger actual model loading by accessing the model
        model_loader.get_model()
        app_state["model_loader"] = model_loader
        app_state["asr_service"] = ASRService(model_loader)
        logger.info("Model loader and ASR service initialized successfully.")
    except ModelLoadError as e:
        logger.error(f"Fatal: Model could not be loaded during application startup: {e}")
        # The app might be unusable.
        app_state["model_loader"] = None
        app_state["asr_service"] = None
        # Depending on policy, you might want the app to exit.
        # For now, /health will report unhealthy.
    except Exception as e:
        logger.error(f"Fatal: An unexpected error occurred during application startup: {e}", exc_info=True)
        app_state["model_loader"] = None
        app_state["asr_service"] = None
    
    yield
    
    logger.info("Application shutdown: Cleaning up resources...")
    # Add any cleanup logic here if needed
    app_state.clear()
    logger.info("Resources cleaned up. Application shutdown complete.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SenseVoice ASR API",
    description="API for FunAudioLLM/SenseVoiceSmall speech-to-text transcription.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Middleware (Optional) ---
# Example: CORS Middleware if your frontend is on a different domain
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Or specify your frontend origin e.g., ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["POST", "GET"], # Added GET for health check
#     allow_headers=["*"]
# )

# --- Exception Handlers ---
@app.exception_handler(AudioProcessingError)
async def audio_processing_exception_handler(request: Request, exc: AudioProcessingError):
    logger.warning(f"Audio processing error for request {request.url.path}: {exc}") # Log as warning for client errors
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=str(exc)).model_dump(),
    )

@app.exception_handler(ModelInferenceError)
async def model_inference_exception_handler(request: Request, exc: ModelInferenceError):
    logger.error(f"Model inference error for request {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail=str(exc)).model_dump(),
    )

@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request: Request, exc: ModelLoadError):
    logger.error(f"Model not available for request {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=503, # Service Unavailable
        content=ErrorResponse(detail="ASR model is not available at the moment. Please try again later.").model_dump(),
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # This handler ensures that FastAPI's own HTTPExceptions are also logged
    # and returned in the standard ErrorResponse format if not already.
    logger.info(f"HTTPException caught: {exc.status_code} - {exc.detail} for request {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail).model_dump(), # Use model_dump() for Pydantic v2+
        headers=exc.headers,
    )

@app.exception_handler(Exception) # Generic fallback for unhandled exceptions
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="An unexpected internal server error occurred.").model_dump(),
    )

# --- API Endpoints ---
@app.post("/asr_pure",
          response_model=ASRResponse,
          summary="Perform Speech-to-Text",
          responses={
              400: {"model": ErrorResponse, "description": "Invalid audio file, format, or processing error"},
              422: {"model": ErrorResponse, "description": "Validation error (e.g., file not provided)"},
              500: {"model": ErrorResponse, "description": "Model inference error or other internal server error"},
              503: {"model": ErrorResponse, "description": "Model not loaded or service unavailable"}
          })
async def asr_pure_endpoint(file: UploadFile = File(..., description="Audio file to be transcribed.")):
    """
    Receives an audio file and returns the ASR transcription.
    Supported formats typically include WAV and MP3, depending on backend processing capabilities.
    """
    if not app_state.get("asr_service"):
        logger.error("ASR service is not available. This indicates a startup failure.")
        # This case should ideally be prevented by a failing health check / deployment strategy
        raise HTTPException(status_code=503, detail="ASR service is not properly initialized. Please check server logs.")

    asr_service: ASRService = app_state["asr_service"]

    # Basic file type check based on filename extension
    # More robust checks (MIME type) could be added if needed,
    # but torchaudio will also validate the content.
    allowed_extensions = (".wav", ".mp3") # Configure as needed
    if not file.filename or not file.filename.lower().endswith(allowed_extensions):
        logger.warning(f"Received file with potentially unsupported name/extension: '{file.filename}'. Allowed: {allowed_extensions}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type or missing filename: '{file.filename}'. Allowed extensions are: {', '.join(allowed_extensions)}"
        )
    
    try:
        transcription, processing_time_ms = await asr_service.transcribe_audio_file(file)
        return ASRResponse(text=transcription, processing_time_ms=processing_time_ms)
    # Specific errors like AudioProcessingError, ModelInferenceError will be caught by dedicated handlers.
    # HTTPException should be re-raised to be handled by FastAPI's mechanisms or our specific handler.
    except HTTPException:
        raise
    except Exception as e: # Fallback for truly unexpected errors within this endpoint's flow
        logger.error(f"Unexpected error in /asr_pure endpoint processing file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during transcription.")

@app.get("/health",
         summary="Health Check",
         response_model=dict)
async def health_check():
    """
    Simple health check endpoint.
    Indicates if the service is running and the ASR model is loaded.
    """
    model_loader = app_state.get("model_loader")
    if model_loader:
        model = model_loader.get_model() # This might raise ModelLoadError if attempted during a failed load
        if model and model != "dummy_model_object":
            return {"status": "ok", "message": "ASR service is healthy and model is loaded."}
        elif model == "dummy_model_object":
            # Still return 200 OK, but indicate degraded state.
            return JSONResponse(
                status_code=200, # Or 503 if you prefer to consider dummy model as unhealthy
                content={"status": "degraded", "message": "ASR service is running with a DUMMY model. Transcription will not work."}
            )
    
    # If model_loader is None or model is None/still dummy after check
    return JSONResponse(
        status_code=503, # Service Unavailable
        content={"status": "unhealthy", "message": "ASR model loader not available or model failed to load."}
    )

# --- __init__.py creation ---
# Remember to create empty __init__.py files in the following directories
# to make them Python packages:
# - app/
# - app/models/
# - app/services/
# Example: touch app/__init__.py app/models/__init__.py app/services/__init__.py

if __name__ == "__main__":
    # This is for local development and debugging only.
    # For production, use Uvicorn directly as shown in deployment_guide.md
    # Example: uvicorn app.main:app --host 0.0.0.0 --port 8888
    import uvicorn
    logger.info("Starting Uvicorn server for local development on http://127.0.0.1:8888")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8888, log_level=LOG_LEVEL.lower(), reload=True)
