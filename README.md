# STTtoLLM — voice chatbot (STT → LLM stage)

The speech-to-text → LLM half of an on-premises voice chatbot. It captures your
speech from the microphone, transcribes it locally with **faster-whisper**, sends
the transcript to an LLM client, and prints the reply **sentence by sentence** as
it streams. Text-to-speech (Kokoro) is **not** in this stage — there's a single,
documented slot left for it.

Out of the box it runs **fully local with a mock LLM**: no network, no API key, no
company laptop. The mock echoes back what Whisper heard so you can verify the STT.

## Pipeline

```
mic ─▶ Silero VAD (endpointing) ─▶ faster-whisper (GPU) ─▶ mock LLM ─▶ printed sentences
```

## Files

| File          | Purpose                                                              |
|---------------|----------------------------------------------------------------------|
| `config.py`   | Every tunable: Whisper model/device, VAD thresholds, API + prompt.   |
| `audio_io.py` | Silero-VAD mic recorder → one 16 kHz mono utterance per call.        |
| `stt.py`      | faster-whisper wrapper → `{text, confident}` with hallucination guards.|
| `llm.py`      | Mock client (default) **and** real company client — same interface.  |
| `main.py`     | The loop; `handle_sentence()` is the single TTS drop-in point.       |

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows
pip install -r requirements.txt
```

### GPU note (common error source)

faster-whisper uses CTranslate2, which on GPU needs the **CUDA cuBLAS and cuDNN**
libraries present on your system. If you see errors like
`Could not load library cudnn_ops64_*.dll` or `cublas64_*.dll not found`:

- Install a **CUDA build of torch** (it bundles the needed runtime libs):
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- Make sure the CUDA / cuDNN DLLs are on your `PATH`.
- Or just set `WHISPER_DEVICE = "cpu"` in `config.py` to sidestep GPU entirely
  while testing — the app auto-falls-back to CPU if the GPU load fails anyway.

## Run (local, with the mock LLM)

```bash
python main.py
```

Speak, then pause. You'll see your transcript and the mock's echoed reply printed
sentence by sentence. No network is used.

## Later: switch to the real company API

When you have API access, it's a **one-line change** plus config:

1. In `llm.py`, flip the swap line at the bottom:
   ```python
   # LLMClient = MockLLMClient
   LLMClient = CompanyLLMClient
   ```
2. Set the endpoint and secret (env vars or `config.py`):
   `LLM_API_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
3. If the real API isn't OpenAI-compatible, adjust the three marked methods in
   `CompanyLLMClient`: `_body()`, `_parse_stream_line()`, `_parse_full_response()`.
4. If it can't stream, set `LLM_SUPPORTS_STREAMING = False` — it then makes one
   full call and splits sentences locally (same yielding interface).

Nothing in `main.py` or any other module needs to change.

## Thresholds to tune on real hardware

All live in `config.py`, flagged `[TUNE]`:

| Setting                 | What it controls                          | When to adjust                                  |
|-------------------------|-------------------------------------------|-------------------------------------------------|
| `TRAILING_SILENCE_MS`   | How long a pause ends your turn (~700 ms). | Cuts you off → raise; feels laggy → lower.       |
| `VAD_THRESHOLD`         | Speech vs. silence sensitivity (0–1).      | Noisy room → raise (~0.6); soft speech missed → lower (~0.35). |
| `WHISPER_MODEL_SIZE`    | Accuracy vs. speed/VRAM (`small` default). | More accuracy & VRAM allows → `medium`; too slow → `base`. |

Also worth tuning: `WHISPER_COMPUTE_TYPE` (`int8_float16` if 4 GB VRAM is tight),
`AVG_LOGPROB_CONFIDENCE_THRESHOLD`, and the `SILENCE_HALLUCINATION_BLOCKLIST`.

## Adding TTS later

`handle_sentence(sentence)` in `main.py` is the one place to synthesize and play
audio. For half-duplex (don't transcribe your own voice), use the existing
`mic.set_muted(True/False)` hook around the `respond()` loop in `main()`.
