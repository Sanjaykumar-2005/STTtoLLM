"""
config.py — single source of truth for every tunable in the STT -> LLM pipeline.

Final deployment is on a different (company) laptop, so EVERYTHING adjustable lives
here: no hard-coded paths, models, thresholds, or secrets in the logic modules.

The three knobs you will most likely retune on real hardware are flagged with
[TUNE]: Whisper model size, the VAD trailing-silence window, and the VAD threshold.
"""

import os


def _load_dotenv():
    """
    Minimal, dependency-free .env loader. If a `.env` file sits next to this file,
    read its KEY=VALUE lines into the environment (without overriding variables that
    are ALREADY set in the real environment — a real $env: var always wins).

    This keeps the endpoint URL and the subscription key out of the committed code
    while saving you from re-exporting them every shell session. Blank lines and
    lines starting with '#' are ignored; surrounding quotes on a value are stripped.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        # utf-8-sig so a BOM (Windows PowerShell's `Set-Content -Encoding utf8`
        # writes one) doesn't get glued onto the first key name.
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

# ======================================================================
# Speech-to-text (faster-whisper)
# ======================================================================

# [TUNE] Model size trades accuracy for speed/VRAM.
#   tiny / base / small / medium / large-v3
#   "small" is a good default on a 4 GB RTX 3050. Bump to "medium" if VRAM allows
#   and accuracy matters more than latency. On CPU, "base" is noticeably faster.
WHISPER_MODEL_SIZE = "small"

# Device + compute precision for CTranslate2 (faster-whisper backend).
#
#   DEV LAPTOP (this machine): "cpu" / "int8".
#     torch here is the CUDA 11.8 build, but faster-whisper's CTranslate2 backend
#     needs CUDA 12 libraries (cublas64_12.dll + cuDNN 9) which aren't installed,
#     so GPU transcription can't run here. CPU works with no extra libs.
#
#   COMPANY LAPTOP (deployment, has CUDA 12 + cuDNN 9): "cuda" / "float16"
#     (or "int8_float16" if 4 GB VRAM is tight). That is the only change needed.
#
#   If "cuda" is set but the CUDA-12 libs are missing, stt.py auto-falls-back to
#   cpu/int8 at startup instead of crashing on the first utterance.
WHISPER_DEVICE = "cpu"          # company laptop: "cuda"
WHISPER_COMPUTE_TYPE = "int8"   # company laptop: "float16"

# Language hint for Whisper. None = auto-detect. Set e.g. "en" to lock it down
# (faster and avoids occasional language flips on short utterances).
WHISPER_LANGUAGE = "en"

# Where faster-whisper caches downloaded models. None = library default
# (~/.cache/huggingface). Set a path here for an offline/locked-down company box.
WHISPER_DOWNLOAD_ROOT = os.environ.get("WHISPER_DOWNLOAD_ROOT") or None

# ----- Hallucination / low-confidence guards (used inside stt.py) -----

# Drop a Whisper segment if it is probably silence.
NO_SPEECH_PROB_THRESHOLD = 0.6      # segment.no_speech_prob above this -> discard
AVG_LOGPROB_DROP_THRESHOLD = -1.0   # segment.avg_logprob below this -> discard entirely

# Confidence flag for the whole utterance. If the mean avg_logprob across kept
# segments is below this, transcription is marked {confident: False} and the main
# loop will echo back what it heard (future: spoken "did you mean...?" confirm).
AVG_LOGPROB_CONFIDENCE_THRESHOLD = -0.7

# Phrases Whisper loves to invent over silence/music. Matched case-insensitively
# against the full stripped transcript; an exact match is dropped as a hallucination.
SILENCE_HALLUCINATION_BLOCKLIST = [
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "see you next time",
    "thanks for watching!",
    "you",
    "bye",
    "bye.",
]

# ======================================================================
# Audio capture + Voice Activity Detection (Silero VAD)
# ======================================================================

SAMPLE_RATE = 16000   # Hz. Whisper and Silero both expect 16 kHz mono.
CHANNELS = 1
AUDIO_DTYPE = "float32"

# Microphone input device. None = system default. To use a specific mic, set this
# to its integer index from the list that diag_record.py / the diagnostic prints
# (e.g. a wired headset or "Realtek HD Audio Mic input" instead of the laptop's
# far-field array mic, which is often very quiet and over-processed).
INPUT_DEVICE = None

# Silero VAD processes fixed-size frames. At 16 kHz the model expects 512 samples
# (= 32 ms) per call. Do not change unless you also change the VAD model contract.
VAD_FRAME_SAMPLES = 512

# [TUNE] Speech probability above which a frame counts as speech (0..1).
#   Raise toward 0.6-0.7 in a noisy room to avoid false triggers.
#   Lower toward 0.3-0.4 in a quiet room if soft speech gets missed.
VAD_THRESHOLD = 0.5

# [TUNE] How much trailing silence ends the turn. Too short = it cuts you off
# mid-thought; too long = laggy. ~700 ms is a comfortable default.
TRAILING_SILENCE_MS = 700

# Keep this much audio from BEFORE speech was detected, so the first phoneme is
# not clipped while VAD was still warming up.
PRE_ROLL_MS = 250

# Hard cap on a single utterance so a stuck-open mic can't record forever.
MAX_UTTERANCE_MS = 20000

# Ignore blips shorter than this as noise (door clicks, coughs) — they won't
# start a real turn.
MIN_SPEECH_MS = 200

# ======================================================================
# LLM — which client is wired up
# ======================================================================

# Default = local mock, no network, no API key. The real company client is built
# in llm.py but NOT used until you flip this (or change the import in main.py).
# Kept here as documentation of the default; main.py imports the mock directly so
# the swap is a genuine one-line import change.
USE_MOCK_LLM = True

# ----- Real company API client settings (only used when wired up later) -----

# OpenAI-compatible /chat/completions endpoint. Company gateway: Azure API Management
# fronting Qwen-32B. The model is selected by the URL path (.../qwen32b/...), not a
# body field — see LLM_MODEL below.
#
# The URL and the key both come from the ENVIRONMENT (or the .env file loaded above),
# never hard-coded here — so the internal endpoint and the secret stay out of git.
# Set them via a `.env` file (copy .env.example) or real env vars:
#   PowerShell:  $env:LLM_API_BASE_URL = "https://.../qwen32b"
#                $env:LLM_API_KEY       = "your-subscription-key"
LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "")
LLM_CHAT_COMPLETIONS_PATH = "/chat/completions"

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# How the key is sent. Azure APIM expects a custom header with no scheme prefix:
#   ocp-apim-subscription-key: <key>
# For a vanilla OpenAI-style endpoint instead, use "Authorization" + "Bearer ".
LLM_API_KEY_HEADER = "ocp-apim-subscription-key"
LLM_API_KEY_PREFIX = ""   # OpenAI-style: "Bearer "

# Model name to send in the request body. The APIM gateway routes by URL path, so
# the working curl omits this entirely — leave it empty and no `model` field is sent.
# Set it (or LLM_MODEL env var) only if a future endpoint requires one.
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# Set False if the real API cannot do SSE streaming. The client then makes one
# full-response call and sentence-splits locally (same yielding interface).
# Default False: the known-good company curl is non-streaming, so this is the safe
# starting point. Flip to True to try SSE streaming (lower perceived latency) once
# you've confirmed the gateway supports `stream: true`.
LLM_SUPPORTS_STREAMING = False

# Network timeouts (seconds). Every call to the real API must be bounded so a dead
# endpoint can't hang the whole loop.
LLM_CONNECT_TIMEOUT = 5.0
LLM_READ_TIMEOUT = 30.0

# Cap reply length. Voice replies should be short; this also bounds latency/cost.
LLM_MAX_TOKENS = 160
LLM_TEMPERATURE = 0.7

# Printed instead of crashing when the LLM errors out or returns nothing.
LLM_FALLBACK_LINE = "Sorry, I didn't catch that. Could you say it again?"

# ======================================================================
# Google Gemini client settings (used by GeminiLLMClient)
# ======================================================================
# Gemini's REST API is shaped differently from OpenAI/APIM (contents+parts, the
# assistant is the "model" role, system text goes in systemInstruction, the key is
# the x-goog-api-key header). GeminiLLMClient in llm.py handles all of that; these
# are its knobs. The generic LLM_MAX_TOKENS / LLM_TEMPERATURE / timeouts / history /
# SYSTEM_PROMPT above are reused.

GEMINI_API_BASE_URL = os.environ.get(
    "GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# API key from the environment / .env, never hard-coded. Get one from Google AI Studio.
#   PowerShell:  $env:GEMINI_API_KEY = "..."   (or put it in .env)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# SSE streaming (:streamGenerateContent?alt=sse). Gemini supports it well, so it's on
# by default for lower perceived latency. Set False to make one full call instead.
GEMINI_STREAM = True

# gemini-2.5-flash is a THINKING model. With a small maxOutputTokens, the thinking
# phase can consume the whole budget and return EMPTY text. 0 disables thinking for
# fast, short voice replies (recommended here). Set None to omit the field (model
# default / dynamic thinking), or a positive int for an explicit thinking budget.
GEMINI_THINKING_BUDGET = 0

# ======================================================================
# Conversation state (owned locally; assume the API is stateless)
# ======================================================================

# How many prior turns (user+assistant pairs) to resend with each request.
# Older turns are truncated. Keep small for voice — long context = slow + costly.
HISTORY_MAX_TURNS = 6

# Voice-optimized system prompt. NO markdown, lists, headings, code, or URLs —
# everything here will eventually be spoken aloud by TTS.
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. You are having a spoken conversation, so "
    "your replies are read aloud. Reply in plain, natural, conversational sentences. "
    "Keep answers to two or three sentences unless the user explicitly asks for more. "
    "Do not use markdown, bullet points, numbered lists, headings, code blocks, "
    "emojis, or URLs. Spell out anything that would be awkward to hear read aloud. "
    "If you are unsure what the user meant, ask a short clarifying question."
)
