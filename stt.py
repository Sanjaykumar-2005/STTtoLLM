"""
stt.py — faster-whisper transcription with hallucination / low-confidence guards.

faster-whisper (CTranslate2 backend) is used instead of vanilla openai-whisper:
same models and accuracy, roughly 4x faster.

transcribe() takes the float32 16 kHz audio that audio_io produced and returns:
    {"text": <cleaned transcript>, "confident": <bool>}

The guards here are what separate a usable transcript from Whisper's habit of
inventing text over near-silence:
  * condition_on_previous_text=False        -> stops self-reinforcing repeat loops,
  * drop segments with high no_speech_prob   -> "that segment was probably silence",
  * drop segments with very low avg_logprob   -> "the model was guessing",
  * a small blocklist of stock silence phrases ("thanks for watching", etc.),
  * a confidence flag from avg_logprob so the caller can confirm what it heard.
"""

import numpy as np

import config

try:
    from faster_whisper import WhisperModel
except Exception as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "faster-whisper is required. Install it with `pip install faster-whisper`. "
        "On GPU it also needs the CUDA cuBLAS and cuDNN libraries — see the README."
    ) from e


def _normalize(text: str) -> str:
    """Lowercase and strip surrounding whitespace/punctuation for blocklist matching."""
    return text.strip().strip(".,!?;:\"' ").lower()


class Transcriber:
    """Loads faster-whisper once and transcribes utterances."""

    def __init__(self):
        self.model = self._load_model()
        # Pre-normalize the blocklist once.
        self._blocklist = {_normalize(p) for p in config.SILENCE_HALLUCINATION_BLOCKLIST}

    def _load_model(self) -> WhisperModel:
        """
        Load the model on the configured device/precision, with graceful fallback.

        cuBLAS/cuDNN problems and tight VRAM are the two most common GPU failures,
        so we step down float16 -> int8_float16 -> CPU int8 instead of crashing.
        """
        attempts = [(config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)]
        if config.WHISPER_DEVICE == "cuda":
            # If the GPU / CUDA-12 libs aren't usable, fall straight back to CPU.
            # We deliberately do NOT auto-retry int8_float16 on the GPU: on a box
            # with broken CUDA libs, a second GPU model-load can HANG instead of
            # erroring. If you want int8_float16 for tight VRAM, set it explicitly
            # via WHISPER_COMPUTE_TYPE in config.py.
            attempts.append(("cpu", "int8"))           # last-resort CPU fallback

        last_err = None
        for device, compute_type in attempts:
            try:
                model = WhisperModel(
                    config.WHISPER_MODEL_SIZE,
                    device=device,
                    compute_type=compute_type,
                    download_root=config.WHISPER_DOWNLOAD_ROOT,
                )
                # Warm up so deferred CUDA-library errors (e.g. a missing
                # cublas64_12.dll, which only loads on the first compute call)
                # surface HERE and trigger fallback, instead of crashing on the
                # user's first real utterance.
                _segments, _ = model.transcribe(
                    np.zeros(config.SAMPLE_RATE, dtype=np.float32),
                    language=config.WHISPER_LANGUAGE,
                    beam_size=1,
                )
                list(_segments)
                if (device, compute_type) != attempts[0]:
                    print(f"[stt] Loaded Whisper on {device}/{compute_type} "
                          f"(fell back from {attempts[0][0]}/{attempts[0][1]}).")
                else:
                    print(f"[stt] Loaded Whisper '{config.WHISPER_MODEL_SIZE}' "
                          f"on {device}/{compute_type}.")
                return model
            except Exception as e:
                last_err = e
                print(f"[stt] Could not load Whisper on {device}/{compute_type}: {e}")

        raise RuntimeError(
            "Failed to load faster-whisper on any device/precision. "
            "Check CUDA cuBLAS/cuDNN install or set WHISPER_DEVICE='cpu' in config."
        ) from last_err

    def transcribe(self, audio: np.ndarray) -> dict:
        """
        Transcribe one utterance. Returns {"text": str, "confident": bool}.
        Empty text + confident=False means "nothing usable was heard".
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return {"text": "", "confident": False}

        segments, _info = self.model.transcribe(
            audio,
            language=config.WHISPER_LANGUAGE,
            beam_size=5,
            condition_on_previous_text=False,  # guard: no repeat-loops
            vad_filter=False,                  # already VAD-gated upstream
        )

        kept_text: list[str] = []
        kept_logprobs: list[float] = []
        for seg in segments:
            # Guard: probably silence.
            if seg.no_speech_prob is not None and \
                    seg.no_speech_prob > config.NO_SPEECH_PROB_THRESHOLD:
                continue
            # Guard: model was guessing.
            if seg.avg_logprob is not None and \
                    seg.avg_logprob < config.AVG_LOGPROB_DROP_THRESHOLD:
                continue
            text = seg.text.strip()
            if text:
                kept_text.append(text)
                if seg.avg_logprob is not None:
                    kept_logprobs.append(seg.avg_logprob)

        full_text = " ".join(kept_text).strip()
        if not full_text:
            return {"text": "", "confident": False}

        # Guard: stock silence-hallucination phrases.
        if _normalize(full_text) in self._blocklist:
            return {"text": "", "confident": False}

        # Confidence from the mean kept avg_logprob.
        if kept_logprobs:
            mean_logprob = sum(kept_logprobs) / len(kept_logprobs)
            confident = mean_logprob >= config.AVG_LOGPROB_CONFIDENCE_THRESHOLD
        else:
            confident = False

        return {"text": full_text, "confident": confident}
