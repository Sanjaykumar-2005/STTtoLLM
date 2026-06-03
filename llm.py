"""
llm.py — two interchangeable LLM clients with one identical interface.

Both clients expose the SAME contract so swapping them is a one-line import change
in main.py and nothing else:

    client = LLMClient()                 # construct once
    for sentence in client.respond(text):  # yields COMPLETE sentences as they form
        handle_sentence(sentence)

Yielding complete sentences (not raw tokens) is the perceived-latency trick: the
future TTS can start speaking sentence 1 while sentence 2 is still generating.

Clients
-------
* MockLLMClient    — LOCAL, no network, no API key. WIRED UP BY DEFAULT. Echoes the
                     transcript back as a 2-3 sentence reply so you can verify what
                     Whisper actually heard while testing on a personal laptop.
* CompanyLLMClient — the real OpenAI-compatible /chat/completions client. Built here
                     but NOT used by default. To go live, change the import in main.py
                     (see the bottom of this file) and fill in the three clearly
                     marked _body() / _parse_stream_line() / _parse_full_response()
                     spots if the real API's shapes differ.

Both own conversation history locally (the API is assumed stateless) and resend the
system prompt + last N turns on every call.
"""

import json
import re
import time

import config

# requests is only needed by the real client. Import lazily-tolerantly so the
# default mock run works even if requests isn't installed yet.
try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# Cosmetic only: per-word delay for the mock's fake stream. 0.0 = instant.
# Bump to e.g. 0.03 if you want to watch sentences appear one at a time.
_MOCK_STREAM_DELAY = 0.0

# Sentence-ending punctuation followed by whitespace or end-of-text.
_SENTENCE_END = re.compile(r"[.!?]+(?:[\"')\]]+)?(\s+|$)")


def stream_sentences(text_chunks):
    """
    Turn a stream of arbitrary text chunks into a stream of COMPLETE sentences,
    emitting each sentence the instant its terminator arrives. Shared by both
    clients: the mock feeds it word-by-word, the real client feeds it SSE deltas.
    """
    buffer = ""
    for chunk in text_chunks:
        if not chunk:
            continue
        buffer += chunk
        while True:
            m = _SENTENCE_END.search(buffer)
            if not m:
                break
            end = m.end()
            sentence = buffer[:end].strip()
            buffer = buffer[end:]
            if sentence:
                yield sentence
    # Flush whatever is left (a final sentence with no trailing punctuation).
    tail = buffer.strip()
    if tail:
        yield tail


class BaseLLMClient:
    """Shared conversation-state handling for both clients."""

    def __init__(self):
        # History excludes the system prompt; stored as alternating user/assistant.
        self.history: list[dict] = []

    def _build_messages(self, user_text: str) -> list[dict]:
        """system prompt + last N turns + the new user turn (older turns truncated)."""
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        keep = config.HISTORY_MAX_TURNS * 2  # each turn = 1 user + 1 assistant msg
        messages.extend(self.history[-keep:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _remember(self, user_text: str, assistant_text: str):
        """Append a completed turn and truncate older history."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        keep = config.HISTORY_MAX_TURNS * 2
        if len(self.history) > keep:
            self.history = self.history[-keep:]

    def respond(self, user_text: str):
        raise NotImplementedError


# ======================================================================
# Mock client — wired up by default. Local, no network, no API key.
# ======================================================================

class MockLLMClient(BaseLLMClient):
    """Echoes the transcript back as a short reply so you can verify Whisper."""

    def respond(self, user_text: str):
        reply = (
            f"I heard you say: {user_text}. "
            "This is the local mock assistant, so no real language model was contacted. "
            "Swap in the company client when the API is available and I will answer for real."
        )
        collected = []
        for sentence in stream_sentences(self._fake_token_stream(reply)):
            collected.append(sentence)
            yield sentence
        self._remember(user_text, " ".join(collected).strip())

    @staticmethod
    def _fake_token_stream(text: str):
        """Emit the reply word-by-word so the sentence aggregator is exercised."""
        for word in text.split(" "):
            if _MOCK_STREAM_DELAY:
                time.sleep(_MOCK_STREAM_DELAY)
            yield word + " "


# ======================================================================
# Real company client — built, but NOT wired up by default.
# ======================================================================

class CompanyLLMClient(BaseLLMClient):
    """
    OpenAI-compatible /chat/completions client with SSE streaming and a
    non-streaming fallback. Every network call is timeout-bounded; on any error
    or empty response it yields a clean fallback line instead of crashing.
    """

    def respond(self, user_text: str):
        if requests is None:
            print("[llm] 'requests' is not installed; cannot reach the company API.")
            yield config.LLM_FALLBACK_LINE
            return

        if not config.LLM_API_BASE_URL or not config.LLM_API_KEY:
            print("[llm] LLM_API_BASE_URL / LLM_API_KEY are not set. Put them in a "
                  ".env file (copy .env.example) or export them as environment "
                  "variables before running.")
            yield config.LLM_FALLBACK_LINE
            return

        messages = self._build_messages(user_text)
        collected = []
        try:
            for sentence in stream_sentences(self._stream_deltas(messages)):
                collected.append(sentence)
                yield sentence
        except Exception as e:
            print(f"[llm] request failed: {e}")
            if not collected:
                yield config.LLM_FALLBACK_LINE
            return  # don't store a failed/partial turn in history

        reply = " ".join(collected).strip()
        if not reply:
            # Empty-response guard: say something rather than going silent.
            yield config.LLM_FALLBACK_LINE
            return
        self._remember(user_text, reply)

    def _stream_deltas(self, messages: list[dict]):
        """Yield text deltas from the API (one big delta when not streaming)."""
        url = config.LLM_API_BASE_URL.rstrip("/") + config.LLM_CHAT_COMPLETIONS_PATH
        headers = {"Content-Type": "application/json"}
        if config.LLM_API_KEY:
            # Header name + prefix are configurable: Azure APIM uses
            # `ocp-apim-subscription-key: <key>` (no prefix); OpenAI uses
            # `Authorization: Bearer <key>`.
            headers[config.LLM_API_KEY_HEADER] = config.LLM_API_KEY_PREFIX + config.LLM_API_KEY
        timeout = (config.LLM_CONNECT_TIMEOUT, config.LLM_READ_TIMEOUT)

        if config.LLM_SUPPORTS_STREAMING:
            with requests.post(
                url, headers=headers,
                json=self._body(messages, stream=True),
                stream=True, timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=False):
                    if not raw:
                        continue
                    delta = self._parse_stream_line(raw)
                    if delta:
                        yield delta
        else:
            # "API cannot stream" path: one full call, split locally.
            resp = requests.post(
                url, headers=headers,
                json=self._body(messages, stream=False),
                timeout=timeout,
            )
            resp.raise_for_status()
            text = self._parse_full_response(resp.json())
            if text:
                yield text

    # ------------------------------------------------------------------
    # >>> ADAPT THESE THREE IF THE REAL API DIFFERS FROM OpenAI-COMPATIBLE <<<
    # ------------------------------------------------------------------

    def _body(self, messages: list[dict], stream: bool) -> dict:
        """
        Request body sent to /chat/completions. Mirrors the known-good company curl
        (messages + max_tokens). `model` and `temperature` are only included when
        configured, since the APIM gateway routes to Qwen by URL and rejects/ignores
        an unexpected `model`. Adjust field names here if a future API differs.
        """
        body = {
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS,
            "stream": stream,
        }
        if config.LLM_MODEL:
            body["model"] = config.LLM_MODEL
        if config.LLM_TEMPERATURE is not None:
            body["temperature"] = config.LLM_TEMPERATURE
        return body

    def _parse_stream_line(self, raw: bytes):
        """
        Pull a text delta out of ONE SSE line. OpenAI-compatible lines look like
        `data: {json}` and the stream ends with `data: [DONE]`. Return None for
        keep-alives / non-content lines.
        """
        try:
            line = raw.decode("utf-8").strip()
        except Exception:
            return None
        if not line or not line.startswith("data:"):
            return None
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return None
        try:
            obj = json.loads(payload)
            return obj["choices"][0]["delta"].get("content")
        except Exception:
            return None

    def _parse_full_response(self, data: dict) -> str:
        """Pull the full reply text out of a non-streaming JSON response."""
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""


# ======================================================================
# Google Gemini client — gemini-2.5-flash via the Generative Language API.
# ======================================================================

class GeminiLLMClient(BaseLLMClient):
    """
    Google Gemini client. Same respond() sentence-stream contract as the others,
    but Gemini's REST shape is different from OpenAI/APIM:
      * messages live in `contents`, each {role, parts:[{text}]},
      * the assistant role is "model" (not "assistant"),
      * the system prompt goes in a top-level `systemInstruction`,
      * the key is the `x-goog-api-key` header,
      * the reply text is candidates[0].content.parts[*].text,
      * max_tokens/temperature live under `generationConfig`.
    Streaming uses `:streamGenerateContent?alt=sse`; both paths reuse stream_sentences.
    """

    def respond(self, user_text: str):
        if requests is None:
            print("[llm] 'requests' is not installed; cannot reach the Gemini API.")
            yield config.LLM_FALLBACK_LINE
            return

        if not config.GEMINI_API_KEY:
            print("[llm] GEMINI_API_KEY is not set. Put it in a .env file (copy "
                  ".env.example) or export it as an environment variable.")
            yield config.LLM_FALLBACK_LINE
            return

        contents = self._build_contents(user_text)
        collected = []
        try:
            for sentence in stream_sentences(self._stream_deltas(contents)):
                collected.append(sentence)
                yield sentence
        except Exception as e:
            print(f"[llm] Gemini request failed: {e}")
            if not collected:
                yield config.LLM_FALLBACK_LINE
            return  # don't store a failed/partial turn in history

        reply = " ".join(collected).strip()
        if not reply:
            yield config.LLM_FALLBACK_LINE
            return
        self._remember(user_text, reply)

    def _build_contents(self, user_text: str) -> list[dict]:
        """history (last N turns) + new user turn, mapped to Gemini's contents shape."""
        keep = config.HISTORY_MAX_TURNS * 2
        contents = []
        for msg in self.history[-keep:]:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        return contents

    def _stream_deltas(self, contents: list[dict]):
        """Yield text deltas from Gemini (one big delta when not streaming)."""
        base = config.GEMINI_API_BASE_URL.rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": config.GEMINI_API_KEY,
        }
        timeout = (config.LLM_CONNECT_TIMEOUT, config.LLM_READ_TIMEOUT)
        body = self._body(contents)

        if config.GEMINI_STREAM:
            url = f"{base}/models/{config.GEMINI_MODEL}:streamGenerateContent"
            with requests.post(
                url, headers=headers, params={"alt": "sse"},
                json=body, stream=True, timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=False):
                    if not raw:
                        continue
                    delta = self._parse_stream_line(raw)
                    if delta:
                        yield delta
        else:
            url = f"{base}/models/{config.GEMINI_MODEL}:generateContent"
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            text = self._extract_text(resp.json())
            if text:
                yield text

    def _body(self, contents: list[dict]) -> dict:
        """Gemini request body: contents + systemInstruction + generationConfig."""
        gen_config = {"maxOutputTokens": config.LLM_MAX_TOKENS}
        if config.LLM_TEMPERATURE is not None:
            gen_config["temperature"] = config.LLM_TEMPERATURE
        if config.GEMINI_THINKING_BUDGET is not None:
            gen_config["thinkingConfig"] = {"thinkingBudget": config.GEMINI_THINKING_BUDGET}
        return {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": config.SYSTEM_PROMPT}]},
            "generationConfig": gen_config,
        }

    def _parse_stream_line(self, raw: bytes):
        """Pull text out of ONE SSE line (`data: {json}`). None for non-content lines."""
        try:
            line = raw.decode("utf-8").strip()
        except Exception:
            return None
        if not line.startswith("data:"):
            return None
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            return self._extract_text(json.loads(payload))
        except Exception:
            return None

    def _extract_text(self, obj: dict):
        """Concatenate the text parts of the first candidate. None if there are none."""
        try:
            parts = obj["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            return text or None
        except Exception:
            return None


# ======================================================================
# THE ONE-LINE SWAP
# ----------------------------------------------------------------------
# main.py / web_ui.py import `LLMClient` from here. Flip the active line below to
# choose a client; set the matching keys/URL in config.py or via .env. Nothing
# else changes — all three clients share the respond() sentence-stream contract.
# ======================================================================
# LLMClient = MockLLMClient       # offline / no-key testing (echoes the transcript)
# LLMClient = CompanyLLMClient    # Qwen-32B via the Azure APIM gateway
LLMClient = GeminiLLMClient       # LIVE: gemini-2.5-flash (Generative Language API)
