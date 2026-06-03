# STTtoLLM — working notes

On-prem voice chatbot, **STT → LLM stage only** (TTS comes later). Mic → Silero VAD
→ faster-whisper → LLM → printed sentences. Default run is fully local with a mock
LLM (no network/API key). Target dev HW: Ryzen 7 + RTX 3050 (4 GB VRAM).

## Files
- `config.py` — single source of all tunables: Whisper model/device/compute, VAD
  thresholds, real-API settings, voice system prompt, history depth. `[TUNE]` flags.
- `audio_io.py` — `MicVAD`: continuous 16 kHz mono stream + Silero VAD endpointing
  (pre-roll, trailing-silence end, max-utterance cap). `record_utterance()` returns
  one float32 utterance. Has no-op `set_muted()` hook for future half-duplex.
- `stt.py` — `Transcriber`: faster-whisper wrapper → `{text, confident}`. Guards:
  `condition_on_previous_text=False`, no_speech_prob / avg_logprob drops, silence
  blocklist, confidence from avg_logprob. Auto device fallback float16→int8_float16→cpu.
- `llm.py` — THREE interchangeable clients, same `respond(text)` yields-sentences
  interface: `MockLLMClient` (offline echo), `CompanyLLMClient` (OpenAI/APIM-shaped,
  Qwen via Azure), `GeminiLLMClient` (gemini-2.5-flash: contents+parts, `model` role,
  systemInstruction, x-goog-api-key, generationConfig, SSE via :streamGenerateContent;
  thinkingBudget=0 so a small max_tokens isn't eaten by thinking). Swap line at bottom
  picks the active one — currently `LLMClient = GeminiLLMClient`.
- `main.py` — the loop. `handle_sentence()` PRINTS now and is the single documented
  TTS drop-in point. Half-duplex mute hooks marked around the `respond()` loop.
- `web_ui.py` — Flask browser front-end (2nd entry point, parallel to main.py; main.py
  untouched). Runs the same blocking STT→LLM loop on a background thread (`Assistant`),
  fans pipeline events to the browser over SSE (`Bus`). Pause/mute reuses
  `mic.set_muted()`. Live settings panel mutates `config` in-memory (VAD threshold read
  live; trailing-silence frames recomputed on the live MicVAD via `audio_io._ms_to_frames`).
  Routes: `/`, `/events` (SSE), `/control/<start|pause|clear>`, `/settings` (GET/POST).
  Run: `python web_ui.py` → http://localhost:5000.
- `templates/index.html` — single self-contained page (inline CSS/JS) for web_ui.py:
  dark chat UI, status pill (loading/idle/listening/transcribing/thinking/error),
  streamed assistant bubbles, low-confidence badge, Start/Pause + Clear, VAD/silence sliders.
- `requirements.txt` — faster-whisper, silero-vad, sounddevice, numpy, requests, torch, flask.
- `README.md` — setup (incl. CUDA cuBLAS/cuDNN gotcha), run-with-mock, one-line swap,
  thresholds to tune.

## Design contracts (don't break)
- Mock ↔ real client are interchangeable via ONE import line in `llm.py`. main.py
  must never need editing to swap.
- All tunables live in `config.py` only — no hard-coded paths/secrets in logic.
- LLM clients yield COMPLETE sentences (perceived-latency for future TTS).
- Whisper is only ever fed VAD-gated speech (avoids silence hallucinations).

## Recent changes
- 2026-05-29: Initial build of the STT→LLM stage — all modules above created from
  scratch (folder was an empty stub).
- 2026-05-29: Verified on dev laptop. faster-whisper GPU path is unusable here —
  torch is the CUDA 11.8 build but CTranslate2 needs CUDA 12 libs (cublas64_12.dll
  + cuDNN 9), missing. So config now uses WHISPER_DEVICE/COMPUTE = cpu/int8
  (company laptop: switch back to cuda/float16). Two stt.py fixes: (1) added a
  startup warmup so a deferred CUDA-lib error surfaces at load and triggers
  fallback instead of crashing on the first utterance; (2) removed the GPU
  int8_float16 auto-retry — on a broken-CUDA box that second GPU load HANGS.
  End-to-end diagnostic passed on CPU: model load 6.7s, Silero + mic + Whisper all
  run; silent capture correctly returns empty text. Real spoken transcript still
  needs a live `python main.py` (interactive mic).
- 2026-05-30: Added a Flask web UI (`web_ui.py` + `templates/index.html`) as a second
  entry point — main.py left untouched so the llm.py mock↔real swap still applies to
  both. Browser ↔ background pipeline thread over SSE; pause/mute via the existing
  set_muted() hook; live VAD-threshold/trailing-silence sliders. Routes smoke-tested via
  Flask test client (GET /, /settings; POST /control/*, /settings all pass; settings
  mutate config in-memory). Full mic→reply path still needs a live run with a real mic.
- 2026-06-01: Wired up the REAL company LLM — Azure APIM gateway fronting Qwen-32B
  (`https://ltceip4prod.azure-api.net/qwen32b/chat/completions`). config.py now points
  there by default; auth is the APIM `ocp-apim-subscription-key` header (new
  `LLM_API_KEY_HEADER`/`LLM_API_KEY_PREFIX` config, so OpenAI-Bearer style still works
  too). `_body()` trimmed to the known-good curl shape (messages + max_tokens; `model`
  only sent when `LLM_MODEL` is set, since APIM routes by URL). `LLM_SUPPORTS_STREAMING`
  defaulted False (curl is non-stream; flip True to try SSE). Flipped the swap line:
  `LLMClient = CompanyLLMClient` is now ACTIVE. Endpoint probed live (401 w/ placeholder
  key — reachable, auth mechanism confirmed); needs the real key via `$env:LLM_API_KEY`
  for a successful call. Key is read from env only — never hard-coded.
- 2026-06-01: Moved BOTH the endpoint URL and the key fully out of the committed code.
  config.py now reads `LLM_API_BASE_URL` + `LLM_API_KEY` from the environment with no
  hard-coded defaults, and a tiny dependency-free `_load_dotenv()` at the top loads a
  local `.env` (real $env: vars still win via setdefault; reads utf-8-sig so a PowerShell
  BOM doesn't break the first key). Added `.env.example` (copy → `.env`) and `.gitignore`
  (ignores `.env`, __pycache__, .venv, utterance.wav). CompanyLLMClient now prints a clear
  "set them in .env / env vars" message instead of silently falling back when either is
  missing. Verified: empty when unset, env var wins, .env loads (incl. BOM case).
- 2026-06-01: Added `GeminiLLMClient` (gemini-2.5-flash via the Generative Language
  API) as a 3rd interchangeable client and flipped the swap to it. New Gemini config
  section in config.py (GEMINI_API_BASE_URL/MODEL/API_KEY[env], GEMINI_STREAM=True,
  GEMINI_THINKING_BUDGET=0 — flash is a thinking model, so 0 stops thinking from eating
  the 160-token cap and returning empty text). Reuses generic LLM_MAX_TOKENS/TEMPERATURE/
  timeouts/SYSTEM_PROMPT/history. web_ui + index.html show "Gemini 2.5 Flash" as the LLM
  mode; .env.example gained GEMINI_API_KEY. Verified: body is valid Gemini JSON (history
  user→user, assistant→model), and the live endpoint accepted the request shape (dummy
  key → API_KEY_INVALID, i.e. only auth failed). Needs a real GEMINI_API_KEY to generate.
