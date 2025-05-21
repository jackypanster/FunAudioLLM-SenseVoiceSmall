import os
import logging
import torch # Placeholder, will be used if model is a PyTorch model
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor # Example if using Hugging Face

logger = logging.getLogger(__name__)

# --- Configuration --- 
# Prefer environment variables with defaults for flexibility
DEFAULT_MODEL_PATH = "/home/llm/model/iic/SenseVoiceSmall"
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
        self.processor = None # If using Hugging Face processor/tokenizer
        self.device = DEVICE
        self._load_model()
        self._initialized = True

    def _load_model(self):
        logger.info(f"Attempting to load SenseVoiceSmall model from: {MODEL_PATH} onto device: {self.device}")
        try:
            # --- Placeholder for actual model loading logic --- 
            # This part needs to be adapted based on how SenseVoiceSmall is actually distributed and used.
            # 
            # Scenario 1: If it's a Hugging Face model
            # self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
            # self.model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH)
            # 
            # Scenario 2: If it's a raw PyTorch model (.pth, .pt, or similar)
            # self.model = torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin")) # Adjust filename
            # # Potentially load a config file for the model architecture if needed
            # # self.model = YourModelClass(*args_from_config)
            # # self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin")))
            # 
            # Scenario 3: If the model directory contains custom loading scripts from FunAudioLLM
            # This might involve importing a specific function or class from their library
            # and calling it with MODEL_PATH.
            # e.g. from funaudiollm.sensevoice import load_model 
            #      self.model = load_model(MODEL_PATH)
            
            # For now, a simple placeholder to simulate a loaded model:
            if not os.path.exists(MODEL_PATH):
                raise ModelLoadError(f"Model path does not exist: {MODEL_PATH}")
            
            # Simulate model object (replace with actual model)
            self.model = "dummy_model_object" # Replace with actual loaded model
            logger.info(f"Successfully loaded a DUMMY model object. PLEASE REPLACE WITH ACTUAL MODEL LOADING.")

            # Example: if self.model is a PyTorch nn.Module:
            # self.model.to(self.device)
            # self.model.eval()
            # logger.info(f"SenseVoiceSmall model successfully loaded to {self.device} and set to eval mode.")

        except FileNotFoundError:
            logger.error(f"Model files not found at {MODEL_PATH}. Please check the path.")
            raise ModelLoadError(f"Model files not found at {MODEL_PATH}.")
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA out of memory while loading model to {self.device}.")
            raise ModelLoadError(f"CUDA out of memory on {self.device}.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model: {e}")

    def get_model(self):
        if self.model is None:
            # This should ideally not happen if lifespan event handles loading properly
            logger.warning("Model was not loaded at init, attempting to load now.")
            self._load_model()
        if self.model is None: # Still None after trying
             raise ModelLoadError("Model is not available.")
        return self.model

    def get_processor(self): # If using Hugging Face processor
        return self.processor

    def get_device(self) -> str:
        return self.device

# Global instance for easy access, managed by FastAPI lifespan or dependency injection
# model_loader = SenseVoiceLoader() # This can be instantiated in main.py lifespan 