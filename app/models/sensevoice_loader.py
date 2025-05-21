import os
import logging
import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor # Commented out
from funasr import AutoModel # Added for SenseVoice

logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_MODEL_PATH = "/home/llm/model/iic/SenseVoiceSmall"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
DEVICE = os.getenv("DEVICE", DEFAULT_DEVICE)

# Determine the model directory for funasr. This could be the HuggingFace identifier or a local path.
# If MODEL_PATH is a local path, funasr should use it directly.
# If MODEL_PATH was intended to be "FunAudioLLM/SenseVoiceSmall" for automatic download,
# that's also supported by funasr's AutoModel.
# For this project, MODEL_PATH is a local directory.
FUNASR_MODEL_NAME_OR_PATH = MODEL_PATH

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
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.model = None
        # self.processor = None # Processor is likely integrated into funasr's model object
        self.device = DEVICE # funasr's AutoModel takes device as an argument
        self._load_model()
        self._initialized = True

    def _load_model(self):
        logger.info(f"Attempting to load SenseVoiceSmall model using funasr from: {FUNASR_MODEL_NAME_OR_PATH} onto device: {self.device}")
        
        # Check if local path exists if FUNASR_MODEL_NAME_OR_PATH is indeed a local path
        if not os.path.isdir(FUNASR_MODEL_NAME_OR_PATH) and "/" in FUNASR_MODEL_NAME_OR_PATH : # A simple check for path-like strings
            logger.error(f"Model path is not a valid directory: {FUNASR_MODEL_NAME_OR_PATH}")
            raise ModelLoadError(f"Model path is not a directory: {FUNASR_MODEL_NAME_OR_PATH}. Ensure it points to the root of the downloaded model.")

        try:
            # Load model using funasr.AutoModel
            # trust_remote_code=True is important if the model directory contains custom model.py files.
            # The FunAudioLLM/SenseVoiceSmall repo on Hugging Face contains a model.py,
            # so this is likely necessary.
            # VAD (Voice Activity Detection) can be integrated here if desired, as shown in funasr examples.
            # For now, focusing on loading the core ASR model.
            # If your model directory /home/llm/model/iic/SenseVoiceSmall contains a 'model.py',
            # you might need remote_code="model.py" or ensure it's picked up automatically.
            # Based on funasr docs, model path itself should be enough if it contains all necessary files.
            # The 'model' parameter in funasr.AutoModel refers to the model name or path.
            self.model = AutoModel(
                model=FUNASR_MODEL_NAME_OR_PATH,
                # model_revision="master", # Optional: if you need a specific revision from HF
                trust_remote_code=True,   # Keep True, funasr might need it for ModelScope models
                device=self.device,
                # vad_model="fsmn-vad", # Optional: VAD model, can be added later
                # vad_kwargs={"max_single_segment_time": 30000}, # Optional: VAD arguments
                # disable_pbar=True # Optional: to disable progress bars during download/load
            )
            
            # funasr's AutoModel typically loads the model and moves it to the specified device.
            # It also handles setting it to eval mode.
            # If the model object has an 'eval' method, it's good practice, but AutoModel might do it.
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'eval'): # Accessing the underlying torch model if nested
                 self.model.model.eval()
            elif hasattr(self.model, 'eval'):
                 self.model.eval()

            logger.info(f"SenseVoiceSmall model successfully loaded using funasr and set to evaluation mode on {self.device}.")

        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError during funasr model loading from {FUNASR_MODEL_NAME_OR_PATH}: {e}", exc_info=True)
            logger.error("Ensure that the model path is correct and all necessary model files are present.")
            raise ModelLoadError(f"Failed to load model from {FUNASR_MODEL_NAME_OR_PATH} due to FileNotFoundError: {e}")
        except ImportError as e: 
             logger.error(f"ImportError during funasr model loading: {e}. Ensure 'funasr' and its dependencies are installed.", exc_info=True)
             raise ModelLoadError(f"Missing libraries for funasr model loading: {e}")
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA out of memory while loading model to {self.device}.")
            raise ModelLoadError(f"CUDA out of memory on {self.device} during model load.")
        except Exception as e: 
            logger.error(f"An unexpected error occurred while loading the model with funasr: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model from {FUNASR_MODEL_NAME_OR_PATH} with funasr due to an unexpected error: {e}")

    def get_model(self):
        if self.model is None:
            logger.error("Attempted to get model, but it's not loaded (likely due to an earlier error).")
            raise ModelLoadError("Model is not available. Check startup logs for loading errors.")
        return self.model

    def get_processor(self):
        # funasr's AutoModel typically integrates the processor.
        # The main model object is used for inference.
        # If a separate processor object is indeed available and needed, this needs adjustment.
        # For now, returning None or the model itself if it acts as its own processor.
        logger.warning("get_processor() called, but funasr.AutoModel usually integrates the processor. Returning the model object or None.")
        # Check if the loaded model object has a 'processor' attribute or similar
        if hasattr(self.model, 'processor'):
            return self.model.processor
        elif hasattr(self.model, 'tokenizer'): # Some models might expose tokenizer
            return self.model.tokenizer
        # If not, the service layer will need to use the model object directly
        return None 

    def get_device(self) -> str:
        return self.device

# Global instance (optional, manage via DI or lifespan in main.py)
# model_loader = SenseVoiceLoader() 