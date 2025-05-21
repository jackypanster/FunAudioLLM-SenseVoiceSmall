import os
import logging
import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperFeatureExtractor # Example for a Whisper-like model

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
        self.processor = None # For Hugging Face models that use a processor/tokenizer/feature_extractor
        self.device = DEVICE
        self._load_model()
        self._initialized = True

    def _load_model(self):
        logger.info(f"Attempting to load SenseVoiceSmall model from: {MODEL_PATH} onto device: {self.device}")
        
        if not os.path.isdir(MODEL_PATH):
            logger.error(f"Model path is not a valid directory: {MODEL_PATH}")
            raise ModelLoadError(f"Model path is not a directory: {MODEL_PATH}. Please ensure it points to the root directory of the downloaded model.")

        try:
            # --- Actual Model Loading Logic ---
            # The following are examples. You MUST adapt this section based on 
            # how FunAudioLLM/SenseVoiceSmall is structured and meant to be loaded.

            # **Option 1: If it's a Hugging Face compatible model (recommended if available)**
            # Check if the model path contains files typical of a Hugging Face pretrained model
            # (e.g., config.json, pytorch_model.bin)
            # try:
            #     logger.info("Attempting to load as a Hugging Face Transformers model...")
            #     # Adjust Auto classes based on SenseVoiceSmall's actual type (e.g., AutoModelForCTC, AutoModelForSpeechSeq2Seq)
            #     # self.processor = AutoProcessor.from_pretrained(MODEL_PATH) 
            #     # Or, if feature extractor and tokenizer are separate:
            #     # self.feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH) # Example
            #     # self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) # Example
            #     self.model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH) # Example: replace with correct AutoModel class
            #     logger.info("Successfully loaded model using Hugging Face Transformers.")
            # except Exception as hf_e:
            #     logger.warning(f"Could not load as a standard Hugging Face model directly from path: {hf_e}. Will try other methods or require manual setup.")
            #     self.model = None # Ensure model is None if HF loading fails

            # **Option 2: If it's a raw PyTorch model (e.g., .pth or .pt checkpoint file)**
            # This usually requires knowing the model class definition.
            # if self.model is None: # Try this if Hugging Face loading didn't work or isn't applicable
            #     logger.info("Attempting to load as a raw PyTorch model...")
            #     # You would need to:
            #     # 1. Define or import your model's class (e.g., `MySenseVoiceModel`)
            #     #    from path.to.model_definition import MySenseVoiceModel
            #     #    model_architecture = MySenseVoiceModel(*args_from_config_if_any)
            #     # 2. Find the checkpoint file.
            #     checkpoint_path = os.path.join(MODEL_PATH, "pytorch_model.bin") # Or actual .pth file name
            #     if not os.path.exists(checkpoint_path):
            #         checkpoint_path = os.path.join(MODEL_PATH, "sensevoice.pth") # Try another common name
            
            #     if os.path.exists(checkpoint_path):
            #         # If you have the model class:
            #         # self.model = model_architecture
            #         # self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            #         # Or, if the .pth file saves the whole model:
            #         # self.model = torch.load(checkpoint_path, map_location=self.device)
            #         logger.info(f"Found PyTorch checkpoint at {checkpoint_path}. You need to integrate the model class definition and state dict loading.")
            #         # **** THIS IS A PLACEHOLDER - YOU MUST IMPLEMENT ACTUAL LOADING ****
            #         raise ModelLoadError(f"Raw PyTorch model found at {checkpoint_path}, but loading logic is not fully implemented. Please define model class and load state_dict.")
            #     else:
            #         logger.warning(f"No common PyTorch checkpoint file found (e.g., pytorch_model.bin, sensevoice.pth) in {MODEL_PATH}")

            # **Option 3: If the model comes with its own loading script/functions from FunAudioLLM**
            # if self.model is None:
            #    logger.info("Checking for custom FunAudioLLM loading scripts/methods...")
            #    # This might involve:
            #    # sys.path.append(MODEL_PATH) # If scripts are in the model dir
            #    # from model_loading_script import load_my_sensevoice_model
            #    # self.model = load_my_sensevoice_model(MODEL_PATH)
            #    # Or similar, based on FunAudioLLM's documentation for SenseVoiceSmall.
            #    pass # Placeholder for this option

            # **** START CRITICAL SECTION: Replace with actual model loading ****
            # For now, as a fallback to prevent crashing if the above are not configured,
            # we'll raise an error indicating real loading is needed.
            # Remove this once you have implemented one of the options above.
            if self.model is None:
                 logger.error("Actual model loading logic is NOT IMPLEMENTED in app/models/sensevoice_loader.py.")
                 logger.error("Please edit the _load_model method to correctly load your SenseVoiceSmall model.")
                 raise ModelLoadError("Model loading logic not implemented. Service cannot start with a real model.")
            # **** END CRITICAL SECTION ****

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"SenseVoiceSmall model successfully loaded to {self.device} and set to evaluation mode.")

        except ModelLoadError: # Re-raise ModelLoadError to be caught by lifespan
            raise
        except FileNotFoundError: # More specific than generic Exception
            logger.error(f"Model-related files not found at {MODEL_PATH}. Please check the path and model content.")
            raise ModelLoadError(f"Model files not found at {MODEL_PATH}.")
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA out of memory while loading model to {self.device}.")
            raise ModelLoadError(f"CUDA out of memory on {self.device} during model load.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model due to an unexpected error: {e}")

    def get_model(self):
        if self.model is None:
            # This might happen if initial loading failed and was caught by lifespan,
            # but something tries to access it again.
            logger.error("Attempted to get model, but it's not loaded (likely due to an earlier error).")
            raise ModelLoadError("Model is not available. Check startup logs for loading errors.")
        return self.model

    def get_processor(self):
        # Return the processor if you are using one (e.g., from Hugging Face)
        # if self.processor is None:
        #    logger.warning("Attempted to get processor, but it's not loaded/used.")
        return self.processor # Might be None if not applicable

    def get_device(self) -> str:
        return self.device

# Global instance for easy access, managed by FastAPI lifespan or dependency injection
# model_loader = SenseVoiceLoader() # This can be instantiated in main.py lifespan 