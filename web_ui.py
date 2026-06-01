"""
web_ui.py — a browser front-end for the STT -> LLM stage.

This is a SECOND entry point that sits alongside main.py; main.py (the plain
terminal loop) is left completely untouched, so the one-line mock<->real client
swap in llm.py keeps working for both. Run it with:

    python web_ui.py

then open http://localhost:5000 in a browser.

Architecture
------------
The pipeline is blocking by nature (mic.record_utterance() waits for a whole
utterance, then transcribe(), then respond() streams sentences). So the loop runs
on ONE background thread (Assistant._run) and every state change is published onto
a Bus. The browser holds a Server-Sent-Events (SSE) connection to /events and just
renders whatever the Bus emits — no websockets, so Flask is the only new dependency.

Pause / mute reuses the half-duplex hook that already exists in audio_io.MicVAD
(set_muted) rather than trying to interrupt the blocking queue read.

Design contracts honoured:
  * main.py is not edited -> the mock<->real swap in llm.py is unaffected.
  * All tunables still live in config.py; the settings panel only mutates those
    in-memory values at runtime (and recomputes the one cached VAD window).
  * LLM clients are used through their existing respond() sentence-stream contract.
"""

import json
import queue
import threading

from flask import Flask, Response, jsonify, render_template, request

import audio_io
import config
from audio_io import MicVAD
from stt import Transcriber
import llm
from llm import LLMClient  # whatever llm.py currently points at (mock by default)


# ======================================================================
# Bus — fan-out of pipeline events to every connected browser (SSE).
# ======================================================================

class Bus:
    """Thread-safe pub/sub. Each SSE connection gets its own Queue subscriber."""

    def __init__(self):
        self._subscribers: "list[queue.Queue]" = []
        self._lock = threading.Lock()

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue"):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: dict):
        with self._lock:
            for q in list(self._subscribers):
                q.put(event)


# ======================================================================
# Assistant — owns the background STT -> LLM loop and current UI state.
# ======================================================================

class Assistant:
    def __init__(self, bus: Bus):
        self.bus = bus
        self.transcriber: "Transcriber | None" = None
        self.client = None
        self.mic: "MicVAD | None" = None

        self.status = "loading"      # loading | idle | listening | transcribing | thinking | error
        self.paused = True           # start paused; user presses Start to listen
        self.transcript: list[dict] = []  # [{role, text, confident?}] for replay on reconnect

        self._thread: "threading.Thread | None" = None
        self._lock = threading.Lock()

    # ----- lifecycle -------------------------------------------------------

    def start(self):
        """Spin up the background loop (loads models, then listens)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _set_status(self, status: str):
        self.status = status
        self.bus.publish({"type": "status", "value": status})

    def _run(self):
        try:
            self._set_status("loading")
            self.bus.publish({"type": "log", "text": "Loading Whisper + Silero VAD..."})
            self.transcriber = Transcriber()
            self.client = LLMClient()
            self.mic = MicVAD()
            self.mic.start()
            self.mic.set_muted(self.paused)  # honour the starting pause state
        except Exception as e:  # model/mic load failure -> surface in the UI
            self._set_status("error")
            self.bus.publish({"type": "log", "text": f"Startup failed: {e}"})
            return

        self.bus.publish({"type": "log", "text": "Ready."})
        self._set_status("idle" if self.paused else "listening")

        while True:
            # While paused, the mic is muted: record_utterance() drains and discards
            # frames and won't return until we unmute, so this call simply parks here.
            self._set_status("idle" if self.paused else "listening")
            audio = self.mic.record_utterance()

            if self.paused:
                # Defensive: if we were paused right as an utterance closed, drop it.
                continue

            # 1) Transcribe with the hallucination / low-confidence guards.
            self._set_status("transcribing")
            result = self.transcriber.transcribe(audio)
            text = result["text"]
            if not text:
                continue  # silence or a dropped hallucination

            self._add_transcript("user", text, confident=result["confident"])

            # 2) Stream the LLM reply, emitting complete sentences as they arrive.
            self._set_status("thinking")
            self.bus.publish({"type": "assistant_start"})
            collected: list[str] = []
            for sentence in self.client.respond(text):
                collected.append(sentence)
                self.bus.publish({"type": "assistant_sentence", "text": sentence})
            self._store_assistant(" ".join(collected).strip())
            self.bus.publish({"type": "assistant_done"})

    # ----- transcript bookkeeping (for reconnect replay) -------------------

    def _add_transcript(self, role: str, text: str, confident: bool = True):
        entry = {"role": role, "text": text}
        if role == "user":
            entry["confident"] = confident
        self.transcript.append(entry)
        self.bus.publish({"type": "user", "text": text, "confident": confident})

    def _store_assistant(self, text: str):
        if text:
            self.transcript.append({"role": "assistant", "text": text})

    # ----- controls (called from Flask request threads) --------------------

    def pause(self):
        self.paused = True
        if self.mic:
            self.mic.set_muted(True)
        self._set_status("idle")

    def resume(self):
        self.paused = False
        if self.mic:
            self.mic.set_muted(False)
        self._set_status("listening")

    def clear(self):
        self.transcript.clear()
        if self.client is not None:
            self.client.history.clear()
        self.bus.publish({"type": "clear"})

    def snapshot(self) -> dict:
        """Everything a freshly-connected browser needs to render current state."""
        return {
            "status": self.status,
            "paused": self.paused,
            "transcript": self.transcript,
        }


# ======================================================================
# Settings — read/update the subset of config.py the UI exposes live.
# ======================================================================

def current_settings() -> dict:
    return {
        "vad_threshold": config.VAD_THRESHOLD,
        "trailing_silence_ms": config.TRAILING_SILENCE_MS,
        "whisper_model": config.WHISPER_MODEL_SIZE,
        "whisper_device": config.WHISPER_DEVICE,
        "whisper_compute": config.WHISPER_COMPUTE_TYPE,
        "llm_mode": "mock" if LLMClient is llm.MockLLMClient else "company",
    }


def apply_settings(data: dict, assistant: Assistant) -> dict:
    """Mutate the live config values the UI is allowed to change."""
    if "vad_threshold" in data:
        # Read live inside record_utterance(), so this takes effect on the next frame.
        config.VAD_THRESHOLD = float(data["vad_threshold"])
    if "trailing_silence_ms" in data:
        ms = int(data["trailing_silence_ms"])
        config.TRAILING_SILENCE_MS = ms
        # MicVAD caches this as a frame count at __init__, so recompute it on the
        # live instance using audio_io's own helper (keeps the conversion identical).
        if assistant.mic is not None:
            assistant.mic.trailing_silence_frames = audio_io._ms_to_frames(ms)
    return current_settings()


# ======================================================================
# Flask app
# ======================================================================

app = Flask(__name__)
bus = Bus()
assistant = Assistant(bus)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/events")
def events():
    """SSE stream. Replays current state on connect, then live events forever."""
    q = bus.subscribe()

    def gen():
        try:
            # Replay so a refresh / new tab restores the conversation + status.
            yield _sse({"type": "snapshot", **assistant.snapshot()})
            while True:
                try:
                    event = q.get(timeout=15)
                    yield _sse(event)
                except queue.Empty:
                    yield ": ping\n\n"  # heartbeat keeps the connection open
        finally:
            bus.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/control/<action>", methods=["POST"])
def control(action: str):
    if action == "start" or action == "resume":
        assistant.resume()
    elif action == "pause":
        assistant.pause()
    elif action == "clear":
        assistant.clear()
    else:
        return jsonify({"error": f"unknown action {action}"}), 400
    return jsonify({"ok": True, "status": assistant.status, "paused": assistant.paused})


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        return jsonify(apply_settings(request.get_json(force=True) or {}, assistant))
    return jsonify(current_settings())


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


if __name__ == "__main__":
    # Kick off model loading + the listen loop before serving so the first browser
    # immediately sees the "loading" status instead of a blank page.
    assistant.start()
    # threaded=True so the long-lived SSE stream doesn't block control/settings
    # requests. use_reloader=False so models aren't loaded twice.
    app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
