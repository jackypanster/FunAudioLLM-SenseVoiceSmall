import logging
import time
import torchaudio
import torch
import io
import asyncio

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
        try:
            self.model = self.model_loader.get_model()
            self.processor = self.model_loader.get_processor()
            self.device = self.model_loader.get_device()
            
            if not self.model or not self.processor:
                # This case should ideally be caught by ModelLoadError in SenseVoiceLoader if critical components are missing
                logger.error("ASRService initialized, but model or processor is missing from loader. This is unexpected if loader claimed success.")
                # Depending on strictness, could raise an error here too.
                # For now, rely on get_model/get_processor to raise if they are None when accessed later.
            else:
                logger.info(f"ASRService initialized successfully with model and processor on device: {self.device}.")

        except ModelLoadError as e:
            logger.error(f"ASRService initialization failed due to ModelLoadError: {e}")
            raise # Re-raise to be handled by application lifespan/startup logic
        except Exception as e:
            logger.error(f"Unexpected error during ASRService initialization: {e}", exc_info=True)
            raise ModelInferenceError(f"ASRService failed to initialize: {e}") # Wrap as ModelInferenceError or a new ServiceInitError

    async def transcribe_audio_file(self, audio_file: UploadFile) -> tuple[str, float]:
        start_time = time.perf_counter()

        if not self.model or not self.processor:
            logger.error("ASRService cannot transcribe: model or processor not available.")
            raise ModelInferenceError("ASR components (model/processor) not loaded. Check service logs.")

        try:
            audio_bytes = await audio_file.read()
            if not audio_bytes:
                raise AudioProcessingError("Uploaded audio file is empty.")

            # 1. Audio Preprocessing
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

            if sample_rate != self.processor.feature_extractor.sampling_rate:
                logger.warning(
                    f"Input audio sample rate ({sample_rate} Hz) differs from processor's expected rate "
                    f"({self.processor.feature_extractor.sampling_rate} Hz). Resampling might occur implicitly "
                    f"by the processor or should be handled explicitly if issues arise."
                )
                # Explicit resampling to what the processor expects (usually 16kHz for many models)
                # If your processor handles resampling, this might be redundant but safe.
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.processor.feature_extractor.sampling_rate)
                waveform = resampler(waveform)
                effective_sample_rate = self.processor.feature_extractor.sampling_rate
            else:
                effective_sample_rate = sample_rate
            
            if waveform.shape[0] != TARGET_CHANNELS: # Ensure mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                else:
                    raise AudioProcessingError(f"Audio has an unexpected number of channels: {waveform.shape[0]}")
            
            # The waveform is kept on CPU here as the processor typically handles moving data to device.
            # If not, you might need: waveform = waveform.to(self.device)
            logger.info(f"Audio preprocessed: duration {waveform.shape[1]/effective_sample_rate:.2f}s, final sample rate {effective_sample_rate}")

        except Exception as e:
            logger.error(f"Error during audio preprocessing for {audio_file.filename}: {e}", exc_info=True)
            raise AudioProcessingError(f"Failed to preprocess audio: {e}")

        # 2. Model Inference
        try:
            # Process audio using the Hugging Face processor
            # The processor prepares features for the model and can also handle moving data to the correct device.
            # Squeezing waveform assuming it's [1, num_samples] for mono audio.
            inputs = self.processor(waveform.squeeze(0).cpu().numpy(), sampling_rate=effective_sample_rate, return_tensors="pt")
            
            # Move inputs to the same device as the model if processor didn't do it (usually it does for PyTorch tensors)
            # Check if inputs need to be manually moved. Some processors return dicts of tensors.
            input_features = inputs.input_features # Or input_values, depending on the processor
            if isinstance(input_features, torch.Tensor):
                input_features = input_features.to(self.device)
            else: # Handle cases where inputs might be a dict of tensors for more complex models
                for key in inputs: # type: ignore
                    if isinstance(inputs[key], torch.Tensor): # type: ignore
                        inputs[key] = inputs[key].to(self.device) # type: ignore
                input_features = inputs # Pass the whole dict if that's what the model expects

            transcription = ""
            with torch.no_grad():
                # Assuming a generate method for sequence-to-sequence models (like Whisper, BART, T5 based ASR)
                # Adjust max_length and other generation parameters as needed for SenseVoiceSmall
                if hasattr(self.model, 'generate'):
                    predicted_ids = self.model.generate(input_features, max_length=256)
                    # Decode the predicted IDs to text using the processor
                    # For some processors, this might be processor.batch_decode or processor.tokenizer.batch_decode
                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                else:
                    # Fallback for models that don't have a .generate() method (e.g. CTC models like Wav2Vec2)
                    # These typically output logits that need an argmax and then decoding by the processor.
                    logger.warning("Model does not have a 'generate' method. Attempting direct call for logits (CTC-like).")
                    if isinstance(input_features, dict):
                        logits = self.model(**input_features).logits
                    else:
                        logits = self.model(input_features).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.batch_decode(predicted_ids)[0]

            if not transcription:
                # This case should ideally not be hit if model.generate or logit processing works.
                logger.error("Transcription resulted in an empty string after model inference.")
                raise ModelInferenceError("Transcription failed: empty result from model.")

            logger.info(f"Transcription successful for {audio_file.filename}.")

        except RuntimeError as e:
            logger.error(f"Runtime error during model inference for {audio_file.filename}: {e}", exc_info=True)
            if "CUDA out of memory" in str(e):
                raise ModelInferenceError("CUDA out of memory during inference. Try a smaller audio file or check GPU.")
            raise ModelInferenceError(f"Model inference runtime error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during model inference for {audio_file.filename}: {e}", exc_info=True)
            raise ModelInferenceError(f"Model inference failed unexpectedly: {e}")

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        return transcription.strip(), processing_time_ms

    # Example helper for a custom PyTorch model (if needed, needs implementation)
    # def _decode_logits(self, logits):
    #     # Placeholder: Implement actual decoding logic (e.g., CTC beam search decoder)
    #     # This depends heavily on the model's output Stunden and architecture.
    #     # For example, if it's a CTC model, you might use torchaudio.functional.ctc_decode
    #     # or a custom decoder.
    #     logger.warning("_decode_logits is a placeholder and not implemented.")
    #     return "Decoded text (placeholder)"

    # def _decode_custom_output(self, model_output):
    #     # Implement this if your model is not a standard HuggingFace one and needs custom decoding
    #     # For example, applying argmax, then mapping indices to characters/tokens.
    #     raise NotImplementedError("Custom model output decoding is not implemented.")

# This import should be at the top, but to avoid breaking the dummy transcription if asyncio isn't used yet:
import asyncio 