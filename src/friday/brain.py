"""Claude: streaming replies sentence by sentence, tools, history, and spend tracking."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Callable

import anthropic

from .tools import ToolRunner

SYSTEM_PROMPT = """\
You are Friday, a voice assistant running on the user's Mac. Everything you write is \
spoken aloud by a text-to-speech voice.

Character
You are calm, precise and efficient: minimal words, maximum clarity. Slightly formal, \
never verbose. You have a subtle, dry wit that you use rarely and with control. Address \
the user as "sir", naturally at the start or end of a sentence, but not in every sentence.

How to answer
- Answer directly. Default to one to four short sentences; expand only when explicitly asked.
- When useful, add one concise recommendation, phrased as "I recommend…".
- If the user's approach is inefficient or incorrect, say so plainly, for example: \
"That approach is suboptimal, sir. I recommend…"
- Use clean, neutral, technical phrasing. Expressions that suit you: "As you wish.", \
"It appears…", "I recommend…". No filler, no repetition, no casual tone.
- When you carry out a command, confirm it briefly ("As you wish, sir.") instead of \
describing what you did.
- Stay understated. Don't be theatrical, don't describe your own personality, and don't \
compare yourself to any fictional assistant, character or film.

Speaking rules
- Plain sentences only: no markdown, lists, headings, emoji, code or URLs unless the user \
explicitly asks for them.
- Write numbers, times, dates and units the way they are said aloud ("twelve degrees", \
"half past three").
- The user's words come from speech recognition and may contain errors. Infer the most \
likely meaning; if it is genuinely unclear, ask one short question.
- Each user message starts with the current local date and time in brackets. Use it for \
anything time related; don't read it back unless asked.
- Before a slow tool such as web search, you may say a short phrase such as "One moment, \
sir." Then answer from the results in a sentence or two, without reading out sources.
- If a tool fails, say so plainly.
- If the user says goodbye, wraps up, or tells you to stop, reply with a very short \
acknowledgement such as "Very good, sir."

Example
User: How do I fix this null pointer crash?
You: It appears the object is used before it is initialised, sir. Add a guard clause \
before the access. I recommend validating inputs at the boundary."""

FILES_PROMPT = """

Projects and files
You can read the user's local projects and documents with read-only tools. When the user \
asks about a project or its code:
- Call project_overview the first time a project comes up in the conversation, then use \
search_files and read_file for only what the question needs. Prefer searching over reading \
whole files.
- Explain by voice: describe what the code does and why, in plain speech. Refer to files by \
name and rough line number ("in wake dot py, around line forty"). Never read code aloud \
symbol by symbol.
- If seeing it would help, offer to open the file; use open_file when the user agrees or asks.
- You can't modify files. If asked for a change, describe precisely what to change and where.
- For longer explanations of code, up to about six sentences is fine.
The user's projects (name: folder):"""

# $ per million tokens: input, output, cache read, 5-minute cache write.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
}
WEB_SEARCH_PRICE = 0.01  # $10 per 1,000 searches

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]?\s+|\n+")
_SOFT_BREAK = re.compile(r"[,;:–—]\s+")


def split_speakable(buf: str, final: bool = False) -> tuple[list[str], str]:
    """Pull complete sentences off the front of buf so TTS can start early."""
    out = []
    while True:
        m = _SENTENCE_END.search(buf)
        if m:
            chunk, buf = buf[: m.end()].strip(), buf[m.end():]
            if chunk:
                out.append(chunk)
            continue
        if len(buf) > 140:  # long clause with no full stop yet: break at a comma
            soft = list(_SOFT_BREAK.finditer(buf, 0, 140))
            if soft:
                cut = soft[-1].end()
                out.append(buf[:cut].strip())
                buf = buf[cut:]
                continue
        break
    if final and buf.strip():
        out.append(buf.strip())
        buf = ""
    return out, buf


class SpendTracker:
    """Persists per-month spend so the monthly budget survives restarts."""

    def __init__(self, path, model: str):
        self.path = path
        self.prices = PRICES.get(model, PRICES["claude-opus-5-5"])
        self._data = json.loads(path.read_text()) if path.exists() else {}

    @property
    def month_key(self) -> str:
        return datetime.now().strftime("%Y-%m")

    @property
    def month_total(self) -> float:
        return self._data.get(self.month_key, 0.0)

    def add(self, usage) -> float:
        p_in, p_out, p_cr, p_cw = self.prices
        cost = (
            (usage.input_tokens or 0) * p_in
            + (usage.output_tokens or 0) * p_out
            + (usage.cache_read_input_tokens or 0) * p_cr
            + (usage.cache_creation_input_tokens or 0) * p_cw
        ) / 1e6
        stu = getattr(usage, "server_tool_use", None)
        if stu is not None:
            cost += (getattr(stu, "web_search_requests", 0) or 0) * WEB_SEARCH_PRICE
        self._data[self.month_key] = self.month_total + cost
        self.path.write_text(json.dumps(self._data, indent=1))
        return cost


class Brain:
    def __init__(self, cfg, tools: ToolRunner, spend: SpendTracker):
        self.cfg = cfg
        self.client = anthropic.Anthropic(max_retries=1, timeout=anthropic.Timeout(30.0, connect=5.0))
        self.tools = tools
        self.spend = spend
        self.history: list[dict] = []
        self._last_activity = 0.0
        self._lock = threading.Lock()
        self._active_stream = None

        system = SYSTEM_PROMPT
        names = [w.title() for w in cfg.wake_words if w.lower() != "friday"]
        if names:
            system += ("\n\nNames\nThe user may call you Friday or " + ", ".join(names)
                       + ". All of these are you; answer to any of them without remarking on it.")
        if tools.files is not None:
            system += FILES_PROMPT + "\n" + tools.files.list_projects()
        if cfg.user_context.strip():
            system += "\n\nAbout the user:\n" + cfg.user_context.strip()
        for path in cfg.context_files:
            p = Path(path).expanduser()
            try:
                system += f"\n\n--- Context file {p.name} ---\n" + p.read_text(errors="replace")[:20000]
            except OSError as e:
                print(f"[warning] context file {p}: {e}")
        # Fixed system + tools so the prefix is cached across every turn.
        # The system breakpoint lets a fresh session reuse the cached prefix; the
        # top-level cache_control on each request caches the growing conversation.
        self._system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self._tools = list(tools.definitions)
        if cfg.web_search:
            self._tools.append({
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": cfg.web_search_max_uses,
            })

    def abort(self):
        """Best effort: close the in-flight HTTP stream so a cancel takes effect immediately."""
        s = self._active_stream
        if s is not None:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass

    def _trim_history(self):
        if time.monotonic() - self._last_activity > self.cfg.session_idle_reset_s:
            self.history.clear()
        # A turn starts at a user message whose content is plain text (not tool results).
        starts = [i for i, m in enumerate(self.history)
                  if m["role"] == "user" and isinstance(m["content"], str)]
        keep = max(self.cfg.history_turns - 1, 0)  # leave room for the turn being added
        if len(starts) > keep:
            cut = starts[-keep] if keep else len(self.history)
            del self.history[:cut]

    def respond(self, text: str, cancel: threading.Event, emit: Callable[[str], None]) -> dict:
        """Stream Claude's reply, calling emit() for each speakable chunk.

        Returns stats: {"cost": float, "ttft": float|None, "cancelled": bool}.
        """
        with self._lock:
            return self._respond(text, cancel, emit)

    def _respond(self, text, cancel, emit) -> dict:
        stats = {"cost": 0.0, "ttft": None, "cancelled": False}
        if self.spend.month_total >= self.cfg.monthly_budget_usd:
            emit("You've reached this month's Claude budget, so I'm paused until next month.")
            return stats

        self._trim_history()
        stamp = datetime.now().strftime("%a %d %b %Y, %H:%M")
        self.history.append({"role": "user", "content": f"[{stamp}] {text}"})
        turn_start = len(self.history)
        spoken: list[str] = []
        t0 = time.monotonic()

        def say(chunk: str):
            if stats["ttft"] is None:
                stats["ttft"] = time.monotonic() - t0
            spoken.append(chunk)
            emit(chunk)

        try:
            for _ in range(10):  # bound the tool loop (file questions can take several lookups)
                buf = ""
                with self.client.messages.stream(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    system=self._system,
                    tools=self._tools,
                    messages=self.history,
                    thinking={"type": "disabled"},  # lowest latency; Haiku 5.5 allows it at effort <= high
                    output_config={"effort": self.cfg.effort},
                    cache_control={"type": "ephemeral"},  # caches the growing conversation too
                ) as stream:
                    self._active_stream = stream
                    for event in stream:
                        if cancel.is_set():
                            break
                        if event.type == "text":
                            chunks, buf = split_speakable(buf + event.text)
                            for c in chunks:
                                say(c)
                    if cancel.is_set():
                        stats["cancelled"] = True
                        break
                    msg = stream.get_final_message()
                self._active_stream = None
                stats["cost"] += self.spend.add(msg.usage)

                chunks, _ = split_speakable(buf, final=True)
                for c in chunks:
                    say(c)

                if msg.stop_reason == "refusal":
                    del self.history[turn_start - 1:]
                    say("Sorry, I can't help with that one.")
                    return stats

                self.history.append({"role": "assistant", "content": msg.content})

                if msg.stop_reason == "pause_turn":  # server tool (web search) paused; resume
                    continue
                tool_uses = [b for b in msg.content if b.type == "tool_use"]
                if msg.stop_reason != "tool_use" or not tool_uses:
                    break

                results = []
                for b in tool_uses:
                    out, is_err = self.tools.run(b.name, b.input)
                    print(f"  [tool] {b.name}({json.dumps(b.input)}) -> {out[:120]}")
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": out, "is_error": is_err})
                self.history.append({"role": "user", "content": results})
                if cancel.is_set():
                    stats["cancelled"] = True
                    break
        except anthropic.APIConnectionError:
            if not cancel.is_set():
                say("I can't reach Claude right now. Check the internet connection.")
            stats["cancelled"] = cancel.is_set()
        except anthropic.AuthenticationError:
            print("  [error] Claude rejected the API key: set ANTHROPIC_API_KEY.")
            say("My Claude API key isn't working, sir.")
        except TypeError as e:  # the SDK raises this when no credentials are configured at all
            if "authentication" not in str(e).lower():
                raise
            print("  [error] No Anthropic API key: export ANTHROPIC_API_KEY.")
            say("I don't have an API key yet, sir.")
        except anthropic.RateLimitError:
            say("I'm being rate limited. Give me a moment.")
        except anthropic.APIStatusError as e:
            print(f"  [error] API error {e.status_code}: {e.message}")
            say("Something went wrong talking to Claude.")
        finally:
            self._active_stream = None
            self._last_activity = time.monotonic()

        if stats["cancelled"]:
            # Drop half-finished tool exchanges; keep what was actually said so the
            # next turn knows where it was cut off.
            del self.history[turn_start:]
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken) + " …"})
        elif self.history and self.history[-1]["role"] == "user" and len(self.history) == turn_start:
            # Errored before any reply: forget the question so history stays tidy.
            del self.history[turn_start - 1:]
        return stats
