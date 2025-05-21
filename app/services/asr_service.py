import logging
import time
import torchaudio
import torch
import io
# import asyncio # asyncio import moved to the bottom as it's not used in this version of the file

from fastapi import UploadFile
from app.models.sensevoice_loader import SenseVoiceLoader, ModelLoadError
from funasr.utils.postprocess_utils import rich_transcription_postprocess # For post-processing funasr output

logger = logging.getLogger(__name__)

# Configuration for ASRService, can be externalized
# TARGET_SAMPLE_RATE = 16000 # funasr model usually defines its expected sample rate
# TARGET_CHANNELS = 1      # funasr model usually defines its expected channels

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
            self.model = self.model_loader.get_model() # This is now the funasr AutoModel instance
            self.device = self.model_loader.get_device()
            
            if not self.model:
                logger.error("ASRService initialized, but model is missing from loader.")
                raise ModelLoadError("ASRService: Model not loaded after SenseVoiceLoader initialization.")
            
            # self.processor is no longer explicitly managed here, as funasr.AutoModel integrates it.
            logger.info(f"ASRService initialized successfully with funasr model on device: {self.device}.")

        except ModelLoadError as e:
            logger.error(f"ASRService initialization failed due to ModelLoadError: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ASRService initialization: {e}", exc_info=True)
            raise ModelInferenceError(f"ASRService failed to initialize: {e}")

    async def transcribe_audio_file(self, audio_file: UploadFile) -> tuple[str, float]:
        start_time = time.perf_counter()

        if not self.model:
            logger.error("ASRService cannot transcribe: funasr model not available.")
            raise ModelInferenceError("ASR components (funasr model) not loaded. Check service logs.")

        try:
            audio_bytes = await audio_file.read()
            if not audio_bytes:
                raise AudioProcessingError("Uploaded audio file is empty.")
            
            # With funasr, we can often pass bytes directly. It handles internal preprocessing.
            # No explicit torchaudio loading and resampling here unless funasr requires a specific format not derived from bytes.
            # The `input` parameter for model.generate can be a file path, bytes, or a NumPy array.
            logger.info(f"Audio received: {audio_file.filename}, size: {len(audio_bytes)} bytes. Passing to funasr model.")

        except Exception as e:
            logger.error(f"Error reading or preparing audio file {audio_file.filename}: {e}", exc_info=True)
            raise AudioProcessingError(f"Failed to read or prepare audio: {e}")

        # Model Inference using funasr
        try:
            # Parameters for model.generate (refer to funasr/SenseVoice documentation for specifics):
            # - input: audio file path, bytes, or numpy array
            # - language: "auto", "zh", "en", etc.
            # - use_itn: bool (Inverse Text Normalization for punctuation, numbers)
            # - batch_size_s: for dynamic batching if sending multiple segments
            # - merge_vad: bool, if VAD is used and segments need merging
            # - cache: for storing intermediate results (e.g., VAD segments)
            # For a single file, many VAD/batching params might not be critical initially.
            
            # Using io.BytesIO to ensure the input is a file-like object if funasr prefers that over raw bytes for some paths.
            # Alternatively, many funasr models accept raw bytes or numpy arrays directly.
            # For safety and wider compatibility within funasr, BytesIO is a good first approach.
            audio_input = io.BytesIO(audio_bytes)

            # Critical: The `input` for `model.generate` in `funasr` expects a file path string,
            # a list of file path strings, a numpy array, or a list of numpy arrays.
            # It does NOT directly take io.BytesIO or raw bytes for the `input` parameter.
            # To use the bytes, we first load it into a waveform with torchaudio, then pass the numpy array.

            waveform, sample_rate = torchaudio.load(audio_input)
            # FunASR expects the audio to be a numpy array.
            # It also expects mono audio. If stereo, average channels.
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            audio_numpy = waveform.squeeze().cpu().numpy()


            res = self.model.generate(
                input=audio_numpy,  # Pass the numpy array of the waveform
                # input_len=torch.tensor([audio_numpy.shape[0]], dtype=torch.int32).to(self.device), # Might be needed by some underlying models
                # fs=sample_rate, # Sampling rate might be needed explicitly by some models if not in processor
                cache={}, # Default cache
                language="auto",  # Or specify, e.g., "zh", "en"
                use_itn=True,    # Apply Inverse Text Normalization
                # batch_size_s=60, # Example, if VAD is used and batching by seconds
                # merge_vad=True, 
                # merge_length_s=15,
            )

            if not res or not isinstance(res, list) or not res[0].get("text"):
                logger.error(f"Transcription failed or returned unexpected format from funasr model for {audio_file.filename}. Result: {res}")
                raise ModelInferenceError("Transcription failed: empty or malformed result from funasr model.")
            
            # FunASR can return rich transcriptions with timestamps, emotions etc.
            # We use rich_transcription_postprocess to get a clean text string.
            transcription = rich_transcription_postprocess(res[0]["text"])

            if not transcription:
                logger.error(f"Post-processed transcription is empty for {audio_file.filename}.")
                # It's possible for audio to have no speech, which is not strictly an error.
                # Depending on requirements, this might be treated differently.
                # For now, we'll return an empty string, but log it.
                transcription = "" # Or raise ModelInferenceError if empty is always an error

            logger.info(f"Transcription successful for {audio_file.filename}.")

        except RuntimeError as e:
            logger.error(f"Runtime error during funasr model inference for {audio_file.filename}: {e}", exc_info=True)
            if "CUDA out of memory" in str(e):
                raise ModelInferenceError("CUDA out of memory during inference. Try a smaller audio file or check GPU.")
            raise ModelInferenceError(f"Funasr model inference runtime error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during funasr model inference for {audio_file.filename}: {e}", exc_info=True)
            raise ModelInferenceError(f"Funasr model inference failed unexpectedly: {e}")

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

# Keep asyncio import for future use or if other parts of the service need it.
import asyncio 