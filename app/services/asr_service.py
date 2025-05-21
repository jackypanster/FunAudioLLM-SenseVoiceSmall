import logging
import time
import torchaudio
import torch
import io
from fastapi import UploadFile
from app.models.sensevoice_loader import SenseVoiceLoader, ModelLoadError # Assuming SenseVoiceLoader is in models directory

logger = logging.getLogger(__name__)

# --- Configuration --- (Can be moved to a config file or env vars)
TARGET_SAMPLE_RATE = 16000 # Example: Most ASR models expect 16kHz
TARGET_CHANNELS = 1      # Example: Mono audio

class AudioProcessingError(Exception):
    "Custom exception for audio processing failures."
    pass

class ModelInferenceError(Exception):
    "Custom exception for model inference failures."
    pass

class ASRService:
    def __init__(self, model_loader: SenseVoiceLoader):
        self.model_loader = model_loader
        # Ensure model is loaded upon service initialization by accessing it
        try:
            self.model = self.model_loader.get_model()
            self.device = self.model_loader.get_device()
            # self.processor = self.model_loader.get_processor() # If using Hugging Face processor
            if self.model == "dummy_model_object": # Check if the dummy is still there
                logger.warning("ASRService initialized with a DUMMY model. Transcription will not work.")
            else:
                logger.info(f"ASRService initialized with model on device: {self.device}")
        except ModelLoadError as e:
            logger.error(f"ASRService initialization failed: Could not load model - {e}")
            # Depending on desired behavior, could re-raise or set a flag indicating service is unhealthy
            raise

    async def transcribe_audio_file(self, audio_file: UploadFile) -> tuple[str, float]:
        start_time = time.perf_counter()

        if self.model_loader.get_model() == "dummy_model_object":
            logger.error("Cannot transcribe: Actual model is not loaded (dummy model in place).")
            raise ModelInferenceError("Transcription service is not properly configured with a real model.")

        try:
            audio_bytes = await audio_file.read()
            if not audio_bytes:
                raise AudioProcessingError("Uploaded audio file is empty.")

            # 1. Audio Preprocessing
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

            # Resample if necessary
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)
                logger.debug(f"Resampled audio from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz.")
            
            # Convert to mono if necessary
            if waveform.shape[0] != TARGET_CHANNELS:
                if waveform.shape[0] > 1: # More than one channel
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                    logger.debug(f"Converted audio to mono. Original channels: {waveform.shape[0]}")
                else: # Should not happen if TARGET_CHANNELS is 1 and audio has 0 channels
                    raise AudioProcessingError(f"Audio has an unexpected number of channels: {waveform.shape[0]}")
            
            # waveform = waveform.to(self.device) # Move tensor to device

            logger.info(f"Audio preprocessed: duration {waveform.shape[1]/TARGET_SAMPLE_RATE:.2f}s, sample rate {TARGET_SAMPLE_RATE}, channels {waveform.shape[0]}")

        except Exception as e:
            logger.error(f"Error during audio preprocessing: {e}", exc_info=True)
            raise AudioProcessingError(f"Failed to preprocess audio: {e}")

        # 2. Model Inference
        try:
            # --- Placeholder for actual model inference --- 
            # This highly depends on the specific SenseVoiceSmall model API.
            # 
            # Example if using Hugging Face model:
            # inputs = self.processor(waveform.squeeze(0).cpu().numpy(), return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE)
            # input_features = inputs.input_features.to(self.device)
            # with torch.no_grad():
            #     predicted_ids = self.model.generate(input_features, max_length=256) # Adjust max_length
            # transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            # 
            # Example if using a custom PyTorch model:
            # waveform_on_device = waveform.to(self.device)
            # with torch.no_grad():
            #    logits = self.model(waveform_on_device) # Model call
            #    # Process logits to get transcription (e.g., CTC decode or argmax if applicable)
            #    transcription = self._decode_logits(logits) # You'd need a _decode_logits method
            
            # For now, a DUMMY transcription:
            logger.warning("Performing DUMMY transcription. PLEASE REPLACE WITH ACTUAL MODEL INFERENCE.")
            # Simulate some processing delay
            await asyncio.sleep(0.1 + waveform.shape[1] / TARGET_SAMPLE_RATE * 0.1) # Simulate 10% of audio duration + 100ms
            transcription = f"Dummy transcription for audio of {waveform.shape[1]/TARGET_SAMPLE_RATE:.2f}s: Hello World."
            # --- End of placeholder ---

            logger.info(f"Transcription successful.")

        except RuntimeError as e: # Catch PyTorch runtime errors, e.g. CUDA errors during inference
            logger.error(f"Runtime error during model inference: {e}", exc_info=True)
            if "CUDA out of memory" in str(e):
                raise ModelInferenceError("CUDA out of memory during inference. Try with a smaller audio file or a more powerful GPU.")
            raise ModelInferenceError(f"Model inference failed with runtime error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during model inference: {e}", exc_info=True)
            raise ModelInferenceError(f"Model inference failed: {e}")

        # 3. Result Postprocessing (Optional)
        # transcription = transcription.strip() # Example

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        return transcription, processing_time_ms

    # Example helper for a custom PyTorch model (if needed, needs implementation)
    # def _decode_logits(self, logits):
    #     # Placeholder: Implement actual decoding logic (e.g., CTC beam search decoder)
    #     # This depends heavily on the model's output Stunden and architecture.
    #     # For example, if it's a CTC model, you might use torchaudio.functional.ctc_decode
    #     # or a custom decoder.
    #     logger.warning("_decode_logits is a placeholder and not implemented.")
    #     return "Decoded text (placeholder)"

# This import should be at the top, but to avoid breaking the dummy transcription if asyncio isn't used yet:
import asyncio 