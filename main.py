"""
main.py — the voice chatbot loop (STT -> LLM stage).

Flow per turn:
    mic + VAD  ->  faster-whisper  ->  (confirm if low confidence)  ->  LLM stream
               ->  handle_sentence(sentence) for each complete sentence

Default run is fully LOCAL: microphone -> Silero VAD -> faster-whisper on GPU ->
MOCK LLM -> printed sentences. No network and no API key required.

    python main.py

To go live with the company API later, change the single import line in llm.py
(LLMClient = CompanyLLMClient) and fill in the API config — nothing in this file
needs to change.
"""

import config
from audio_io import MicVAD
from stt import Transcriber
from llm import LLMClient  # the one-line swap point lives in llm.py


def handle_sentence(sentence: str):
    """
    =====================================================================
    THE SINGLE TTS DROP-IN POINT.
    =====================================================================
    Called once per COMPLETE sentence as the LLM produces it. Right now it just
    prints. When the TTS stage is added (Kokoro), this is the ONE place to:
        1. synthesize `sentence` to audio, and
        2. play it.
    For half-duplex playback, mute the mic while speaking using the hook that
    already exists in audio_io.MicVAD (mic.set_muted(True/False)) — see the note
    around the respond() loop in main() below. Everything upstream already yields
    sentences one at a time so sentence 1 can be spoken while sentence 2 generates.
    """
    print(f"  {sentence}")


def main():
    print("Loading models (this can take a moment on first run)...")
    transcriber = Transcriber()
    client = LLMClient()

    print("\nReady. Start speaking — pause when you're done. Press Ctrl+C to quit.\n")

    with MicVAD() as mic:
        while True:
            # 1) Capture one VAD-gated utterance (blocks until you speak + pause).
            audio = mic.record_utterance()

            # 2) Transcribe with hallucination/low-confidence guards.
            result = transcriber.transcribe(audio)
            text = result["text"]
            if not text:
                # Empty = silence or a dropped hallucination. Just keep listening.
                continue

            print(f"You: {text}")

            # 3) Low-confidence handling: echo what we heard so the user can see it.
            #    (Later this becomes a spoken "did you mean ...?" confirmation.)
            if not result["confident"]:
                print(f"  [low confidence — I heard: \"{text}\"]")

            # 4) Stream the LLM reply and emit complete sentences as they arrive.
            print("Assistant:")
            # --- future half-duplex: mic.set_muted(True) here ---
            for sentence in client.respond(text):
                handle_sentence(sentence)
            # --- future half-duplex: mic.set_muted(False) here ---
            print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
