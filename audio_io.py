"""
audio_io.py — microphone capture with Silero-VAD endpointing.

Runs a continuous 16 kHz mono float32 input stream and hands back ONE complete
spoken utterance at a time. Voice-activity detection decides when the user starts
and stops talking, so Whisper is only ever fed real speech (silence in -> silence
hallucinations out).

Endpointing rules (all tunable in config.py):
  * a turn STARTS after a short run of speech frames (debounced so a cough/click
    doesn't trigger it),
  * a short PRE-ROLL of audio from just before the trigger is prepended so the
    first phoneme isn't clipped,
  * a turn ENDS after TRAILING_SILENCE_MS of quiet, or once MAX_UTTERANCE_MS is hit.

A `muted` flag is included now as a no-op (nothing sets it this stage). It is the
hook for half-duplex operation later: when TTS is speaking we will mute the mic so
the assistant doesn't transcribe its own voice.
"""

import math
import queue
from collections import deque

import numpy as np

import config

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "sounddevice is required for microphone capture. Install it with "
        "`pip install sounddevice` (and ensure a working PortAudio backend)."
    ) from e

try:
    import torch
    from silero_vad import load_silero_vad
except Exception as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "silero-vad and torch are required for voice activity detection. "
        "Install them with `pip install silero-vad torch`."
    ) from e


def _ms_to_frames(ms: int) -> int:
    """Convert a duration in ms to a whole number of VAD frames (rounding up)."""
    frame_ms = config.VAD_FRAME_SAMPLES / config.SAMPLE_RATE * 1000.0
    return max(1, math.ceil(ms / frame_ms))


class MicVAD:
    """Continuous mic listener that yields one VAD-gated utterance per call."""

    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self.frame_samples = config.VAD_FRAME_SAMPLES

        # Endpointing windows expressed in whole VAD frames.
        self.trailing_silence_frames = _ms_to_frames(config.TRAILING_SILENCE_MS)
        self.pre_roll_frames = _ms_to_frames(config.PRE_ROLL_MS)
        self.max_utterance_frames = _ms_to_frames(config.MAX_UTTERANCE_MS)
        self.onset_frames = _ms_to_frames(config.MIN_SPEECH_MS)

        # Mute hook for future half-duplex playback. No-op this stage.
        self.muted = False

        # Silero VAD model (loaded once; keeps internal recurrent state).
        self._model = load_silero_vad()

        # Audio frames flow from the sounddevice callback into this queue.
        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    # ----- stream lifecycle -------------------------------------------------

    def start(self):
        """Open and start the input stream."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=config.CHANNELS,
            dtype=config.AUDIO_DTYPE,
            blocksize=self.frame_samples,   # one VAD frame per callback
            device=config.INPUT_DEVICE,     # None = system default mic
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        """Stop and close the input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ----- internals --------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice thread: push a flat float32 copy of each block onto the queue."""
        # `status` carries xrun/overflow warnings; we deliberately ignore them so a
        # transient glitch doesn't kill the stream.
        self._frame_queue.put(indata[:, 0].copy())

    def _next_frame(self) -> np.ndarray:
        """Block until the next audio frame is available."""
        return self._frame_queue.get()

    def _speech_prob(self, frame: np.ndarray) -> float:
        """Run one frame through Silero VAD and return P(speech)."""
        # Silero at 16 kHz expects exactly frame_samples; pad/truncate defensively.
        if len(frame) != self.frame_samples:
            buf = np.zeros(self.frame_samples, dtype=np.float32)
            n = min(len(frame), self.frame_samples)
            buf[:n] = frame[:n]
            frame = buf
        with torch.no_grad():
            return self._model(torch.from_numpy(frame), self.sample_rate).item()

    # ----- public API -------------------------------------------------------

    def record_utterance(self) -> np.ndarray:
        """
        Block until the user speaks a complete utterance, then return it as a
        1-D float32 numpy array at config.SAMPLE_RATE. Loops past noise blips and
        muted spans, so the return value is always real speech.
        """
        # Reset recurrent VAD state so the previous turn doesn't bias this one.
        self._model.reset_states()

        pre_roll: "deque[np.ndarray]" = deque(maxlen=self.pre_roll_frames)
        speech_frames: list[np.ndarray] = []
        triggered = False
        consecutive_speech = 0
        silence_frames = 0

        while True:
            frame = self._next_frame()

            # Muted (future half-duplex): drain and discard, never trigger a turn.
            if self.muted:
                pre_roll.clear()
                triggered = False
                consecutive_speech = 0
                speech_frames.clear()
                continue

            is_speech = self._speech_prob(frame) >= config.VAD_THRESHOLD

            if not triggered:
                pre_roll.append(frame)
                if is_speech:
                    consecutive_speech += 1
                    if consecutive_speech >= self.onset_frames:
                        # Confirmed speech onset. Seed the buffer with the pre-roll
                        # (which already includes these onset frames).
                        triggered = True
                        speech_frames.extend(pre_roll)
                        silence_frames = 0
                else:
                    consecutive_speech = 0
                continue

            # --- triggered: accumulating the utterance ---
            speech_frames.append(frame)
            if is_speech:
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= self.trailing_silence_frames:
                    break  # trailing silence -> end of turn

            if len(speech_frames) >= self.max_utterance_frames:
                break  # safety cap so a stuck mic can't record forever

        return np.concatenate(speech_frames).astype(np.float32)

    # ----- mute hook (future half-duplex) -----------------------------------

    def set_muted(self, muted: bool):
        """Toggle the mute flag. No-op effect this stage; wired for TTS playback."""
        self.muted = bool(muted)
