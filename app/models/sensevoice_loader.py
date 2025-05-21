import os
import logging
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor # Using common classes for ASR
# If SenseVoiceSmall uses a different base model, e.g., Wav2Vec2-type CTC, you might need:
# from transformers import AutoModelForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)

# --- Configuration --- 
# Prefer environment variables with defaults for flexibility
DEFAULT_MODEL_PATH = "/home/llm/model/iic/SenseVoiceSmall" # From pro.md
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
DEVICE = os.getenv("DEVICE", DEFAULT_DEVICE)

class ModelLoadError(Exception):
    "Custom exception for model loading failures."
    pass

class SenseVoiceLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SenseVoiceLoader, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # Ensure __init__ is called only once for the singleton
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.model = None
        self.processor = None
        self.device = DEVICE
        self._load_model()
        self._initialized = True

    def _load_model(self):
        logger.info(f"Attempting to load SenseVoiceSmall model from: {MODEL_PATH} onto device: {self.device}")
        
        if not os.path.isdir(MODEL_PATH):
            logger.error(f"Model path is not a valid directory: {MODEL_PATH}")
            raise ModelLoadError(f"Model path is not a directory: {MODEL_PATH}. Ensure it points to the root of the downloaded Hugging Face model.")

        try:
            # --- Attempting to load as a Hugging Face Transformers model ---
            # This assumes SenseVoiceSmall is compatible with AutoProcessor and AutoModelForSpeechSeq2Seq.
            # If it's based on another architecture (e.g., Wav2Vec2 CTC), you'll need to adjust these classes.
            logger.info(f"Loading Hugging Face processor from {MODEL_PATH}...")
            self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
            logger.info(f"Successfully loaded processor. Loading model from {MODEL_PATH}...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH)
            logger.info("Successfully loaded model using Hugging Face Transformers.")
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"SenseVoiceSmall model transferred to {self.device} and set to evaluation mode.")

        except OSError as e: # Catches errors like model not found, config not found at path
            logger.error(f"OSError during Hugging Face model loading from {MODEL_PATH}: {e}", exc_info=True)
            logger.error("Ensure that MODEL_PATH contains all necessary Hugging Face model files (config.json, pytorch_model.bin, tokenizer files, etc.).")
            raise ModelLoadError(f"Failed to load Hugging Face model from {MODEL_PATH} due to OSError: {e}")
        except ImportError as e: # If transformers or a dependency is missing
             logger.error(f"ImportError during Hugging Face model loading: {e}. Ensure 'transformers' and 'sentencepiece' (and other deps) are installed.", exc_info=True)
             raise ModelLoadError(f"Missing libraries for Hugging Face model loading: {e}")
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA out of memory while loading model to {self.device}.")
            raise ModelLoadError(f"CUDA out of memory on {self.device} during model load.")
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"An unexpected error occurred while loading the Hugging Face model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model from {MODEL_PATH} due to an unexpected error: {e}")

    def get_model(self):
        if self.model is None:
            # This might happen if initial loading failed and was caught by lifespan,
            # but something tries to access it again.
            logger.error("Attempted to get model, but it's not loaded (likely due to an earlier error).")
            raise ModelLoadError("Model is not available. Check startup logs for loading errors.")
        return self.model

    def get_processor(self):
        if self.processor is None:
            logger.warning("Attempted to get processor, but it's not loaded. This might be an issue if the model requires it.")
        return self.processor

    def get_device(self) -> str:
        return self.device

# Global instance for easy access, managed by FastAPI lifespan or dependency injection
# model_loader = SenseVoiceLoader() # This can be instantiated in main.py lifespan 