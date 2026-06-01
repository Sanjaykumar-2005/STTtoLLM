"""diag_record.py — record ONE real utterance through the pipeline and dump
everything needed to diagnose bad transcriptions.

Run it, speak ONE clear sentence, then pause ~1 second. Paste the output here,
and play utterance.wav to hear exactly what Whisper was given."""

import wave
import numpy as np

import config
from audio_io import MicVAD
from stt import Transcriber

print("Loading model...")
tx = Transcriber()

print("\n>>> Speak ONE sentence now, then pause. <<<\n")
with MicVAD() as mic:
    audio = mic.record_utterance()

dur = len(audio) / config.SAMPLE_RATE
rms = float(np.sqrt(np.mean(audio ** 2)))
peak = float(np.max(np.abs(audio)))
print(f"Captured {dur:.2f}s   RMS={rms:.4f}   peak={peak:.3f}")
if peak < 0.05:
    print("  !! Very quiet — mic gain is likely too low. This alone wrecks accuracy.")
if dur < 0.8:
    print("  !! Very short — VAD may be cutting you off (raise TRAILING_SILENCE_MS).")

# Save what Whisper actually received so you can listen to it.
pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
with wave.open("utterance.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(config.SAMPLE_RATE)
    w.writeframes(pcm16.tobytes())
print("  -> saved utterance.wav  (play it: does it sound like what you said?)")

# Raw transcription with AUTO language detection + per-segment confidence.
print("\n--- raw faster-whisper, AUTO language detect ---")
segments, info = tx.model.transcribe(
    audio, language=None, beam_size=5, condition_on_previous_text=False,
)
print(f"detected language: {info.language}  (p={info.language_probability:.2f})")
for s in segments:
    print(f"  [{s.start:5.2f}-{s.end:5.2f}] "
          f"no_speech={s.no_speech_prob:.2f} avg_logprob={s.avg_logprob:.2f} "
          f"text={s.text!r}")

# Does simple software gain help, or is it a capture/SNR problem? Amplify the
# waveform to a healthy peak and re-transcribe. If THIS reads correctly, gain
# helps; if it's still wrong, the recording itself is too noisy/unclear and you
# need a better mic / louder, closer speech.
peak_now = float(np.max(np.abs(audio))) or 1.0
gain = min(0.5 / peak_now, 30.0)
amp = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
print(f"\n--- amplified x{gain:.1f} (peak -> {float(np.max(np.abs(amp))):.2f}) ---")
segs2, _ = tx.model.transcribe(
    amp, language=config.WHISPER_LANGUAGE, beam_size=5,
    condition_on_previous_text=False,
)
for s in segs2:
    print(f"  avg_logprob={s.avg_logprob:.2f}  text={s.text!r}")

# What the guarded Transcriber returns (language locked to config).
print(f"\n--- guarded Transcriber (language='{config.WHISPER_LANGUAGE}') ---")
print(tx.transcribe(audio))
