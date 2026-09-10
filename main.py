"""
question and answer in the same language 

============================================================
  🤖 Speech-to-Speech AI Chatbot — Powered by Sarvam AI
============================================================
  Stack:
    STT  → Sarvam Saaras (saaras:v4)
    LLM  → Sarvam-105B Conversations (streamed, sentence-by-sentence)
    TTS  → Sarvam Bulbul v3 (streamed + cached)
    Offline fallback → espeak (no internet required)

  Language Support:
    → Speak English → RoboBot replies & speaks in English
    → Speak Hindi   → RoboBot replies & speaks in Hindi
    → Switches instantly every message — no confusion

  Language Detection (3-layer):
    1. Sarvam STT language detection (fast, sometimes wrong)
    2. Script scan of transcript  (ground truth — never lies)
    3. Default → English

  Optimisations in this version:
    1. Streaming input  — speech is captured continuously and each
       sentence-sized chunk (separated by a short mid-turn pause) is
       transcribed in the background WHILE the user keeps talking.
    2. Streaming output — the LLM reply is streamed token-by-token;
       each finished sentence is sent to TTS and played immediately
       while the next sentence is still being generated/synthesised.
    3. Prompt caching   — (a) exact-match reply cache so a repeated
       question skips the LLM call entirely, (b) a disk-backed audio
       cache so a phrase is never re-synthesised twice, and (c) the
       system+history prefix is kept stable so Sarvam's own server-side
       prefix caching can kick in.
    4. Empty strings are filtered out at every hand-off point before
       they ever reach an API (STT segment, combined transcript,
       individual streamed sentences).
    5/6/7. Layered, offline-safe error handling — see `handle_error()`.

  State Machine:
    IDLE ──(wake word)──► LISTENING ──(speech)──► THINKING ──► SPEAKING
      ▲                        │                                  │
      └──────(10s silence)─────┘◄─────────────────────────────────┘

  Wake word: "Hello" / "Hey"
============================================================
"""

import os
import io
import re
import time
import queue
import socket
import hashlib
import threading
import subprocess
from enum import Enum
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
from dotenv import load_dotenv

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise RuntimeError("SARVAM_API_KEY is not set. Add it to your .env file.")

# Sarvam Saaras v4 speech-to-text model.
STT_MODEL  = "saaras:v4"
# Sarvam-hosted chat model.
CHAT_MODEL = "sarvam-105b-conversations"

# Sarvam Bulbul v3 TTS
# "priya" is a production-recommended female voice for Hindi and also
# works well across English/Indic languages. Speaker names are lowercase.
TTS_MODEL = "bulbul:v3"
TTS_VOICE_EN = "roopa"
TTS_VOICE_HI = "roopa"
TTS_SAMPLE_RATE = 24000
TTS_PACE = 1.0
TTS_TEMPERATURE = 0.6

SAMPLE_RATE = 16000
CHANNELS    = 1
# Ceiling on reply length. Not meant to force short answers -- it's just a
# backstop so a rare runaway generation can't blow the latency budget. 350
# gives a real multi-part answer room to finish naturally (Devanagari tends
# to use more tokens per word than English) without slowing down short
# replies at all -- the model still stops at finish_reason="stop" on its
# own; this ceiling only matters for answers that actually needed the room.
MAX_TOKENS  = 80
# Keep only the most recent N user/assistant turn-pairs per language in the
# prompt. Without this, history grows for as long as the bot stays awake at
# an event (hours), which slowly inflates every future prompt and therefore
# every future "LLM First Token" latency -- exactly the kind of thing that
# looks fine in a quick test and then creeps past the 2.5s budget by evening.
MAX_HISTORY_TURNS = 6

# ── VAD tuning ─────────────────────────────────────────────
# Fallback / initial value — overwritten at startup by calibrate_noise_floor()
# once the mic's actual ambient noise level is known (airports, receptions,
# and stage events all have very different noise floors, so a single fixed
# constant was previously a real accuracy risk: too low → false triggers on
# crowd noise, too high → soft/far speech never crosses it and gets dropped).
ENERGY_THRESHOLD     = 0.10
MIN_ENERGY_THRESHOLD = 0.03   # never calibrate below this (near-silent rooms)
MAX_ENERGY_THRESHOLD = 0.25   # never calibrate above this (very loud rooms)
# Was 1.2s. Every turn pays this in full before STT even starts, so it is
# one of the biggest single latency knobs. 0.8s is a common sweet spot for
# conversational assistants -- if users start getting cut off mid-thought,
# raise it back up in 0.1s steps; if RoboBot still feels slow to react and
# nobody is getting cut off, it can go a little lower (try 0.6-0.7s).
SILENCE_AFTER_SPEECH = 1.0
# Was 0.45s. Governs how quickly a mid-turn chunk gets shipped to Saaras
# in the background. Lower = STT gets a head start sooner, at the cost of
# slightly choppier segment boundaries. 0.35s tested fine; recalibrate if
# segments start splitting mid-word too often.
SENTENCE_PAUSE       = 0.35
# Adaptive silence detection: once at least one mid-turn segment has
# already been flushed to the STT pool (see `any_mid_flush` below), the
# background transcription for the bulk of the utterance has a head
# start, so we don't need to wait as long to conclude the turn -- only
# the short trailing remainder is still on the critical path.
# Was 0.6s — that's *less* total silence than a single natural mid-question
# pause (0.35s SENTENCE_PAUSE + only 0.25s more), so the bot could cut
# a user off right after they paused to think, before finishing their
# question. 0.75s keeps almost all of the latency win while giving people
# real breathing room to complete a thought — accuracy matters more than
# the ~150ms saved here.
SILENCE_AFTER_SPEECH_FOLLOWUP = 0.75
PRE_ROLL_CHUNKS      = 6
# A captured segment shorter than this is rejected LOCALLY, before it's ever
# sent to the Sarvam Speech-to-Text API -- a noise burst used to still get shipped
# and you paid that round-trip for nothing. Only real risk: a genuinely
# short single word (e.g. "haan") spoken in under 0.5s -- worth testing,
# but that's threshold tuning, not a latency trade-off.
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1

IDLE_TIMEOUT      = 50.0
IDLE_POLL_TIMEOUT = 40.0

# ── Perf / debug toggles ───────────────────────────────────
# Print the per-turn latency breakdown (Speech End → STT → GPT First
# Token → Sentence Complete → TTS Start → Playback Started → Total).
# Safe to leave on in production; it's just terminal output.
PRINT_LATENCY_TIMINGS = True
# Stream the FIRST spoken sentence of every turn straight into `mpg123`'s
# stdin as Sarvam Bulbul v3 produces it, instead of waiting for the whole
# sentence to finish synthesising before playback starts. This is the
# single biggest win for "time to first sound" because every later
# sentence already overlaps its synthesis with the previous sentence's
# playback (see StreamingSpeaker) -- only sentence #1 has nothing to
# overlap with. Requires `mpg123` (sudo apt install mpg123). If it's
# missing, or streaming fails for any reason, RoboBot automatically
# falls back to the original synth-then-play path, so this is safe to
# leave on even if mpg123 isn't installed yet.
STREAM_TTS_PLAYBACK = True
# Diagnostic only -- times the pieces INSIDE transcribe_segment() (WAV
# encode, the actual Sarvam API call, and whether the SDK silently retried
# after a transient failure) and prints them per segment. The mystery
# 5000ms-ish STT spikes aren't explained by connection setup (IPv6/IPv4
# connect both came back under 60ms), so this narrows down whether the
# time is Sarvam's response itself or a retry+backoff. Off by default --
# turn on only while chasing this, then back off; it's a couple of
# time.time() calls plus a print, negligible overhead either way.
DEBUG_STT_TIMING = False

WAKE_WORDS = ["hello", "hey"]

SYSTEM_EN = (
    "Your name is Aayro. You are a helpful AI assistant created by Robotwala. "
    "Answer the user's full question directly and accurately — never ignore part of "
    "a multi-part question. "
    "you are a female robot answer the questions accordingly."
    "Keep replies natural and conversational, as long as they need to be to "
    "actually answer the question well — don't pad, but don't cut yourself short either. "
    "No bullet points or markdown, no filler like 'great question'. "
    "The user is speaking English in this conversation, so you must always respond in "
    "English, in the Latin script only — never switch to Hindi or Devanagari script, "
    "even if a word or phrase in the user's message happens to be a Hindi/Urdu loanword "
    "or name. "
    "Keep your full answer under roughly 150 words — you have a hard output limit, so "
    "finish your thought rather than running long."
)

SYSTEM_HI = (
    "Aapka naam Aayro hai. Aap Robotwala dwara banaya gaya helpful AI assistant hain. "
    "User ke poore sawaal ka seedha aur sahi jawab dein — agar sawaal ke kai hisse hain "
    "to kisi bhi hisse ko nazarandaz mat karein. "
    "tum ek female ho , usko hisaab se answer krna ."
    "Jawab natural aur batcheet ke andaz mein dein — sawaal ka sahi jawab dene ke liye "
    "jitna zaroori ho utna lamba rakhein, na zyada padding karein na jawab ko jabardasti chhota karein. "
    "Koi bullet points ya markdown nahi, 'great question' jaisa filler nahi. "
    "Agar user ki language Hindi hai, to hamesha Hindi mein sirf Devanagari script ka use "
    "karke reply do. Agar user Hindi ko Roman Hindi ya Urdu (Perso-Arabic) script mein likhe, "
    "tab bhi hamesha Hindi ki Devanagari script mein hi jawab do. Kabhi bhi Urdu (Perso-Arabic) "
    "script mein reply mat dena. "
    "Apna poora jawab lagbhag 150 shabdon ke andar rakhein — aapki ek hard output limit hai, "
    "isliye jawab ko beech mein chhodne ke bajaye poora karke khatam karein."
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।\n])\s+")
# Used ONLY to flush the very first spoken chunk of a turn. Splitting on a
# clause-level comma too (in addition to sentence-enders) means playback can
# start as soon as the first clause is ready instead of waiting for the
# model to finish a whole first sentence -- this is the single biggest lever
# left for hitting a sub-2.5s "silence → first sound" budget on longer
# replies. Every sentence after the first still uses the normal
# SENTENCE_SPLIT_RE so the reply doesn't sound choppy throughout.
EARLY_SPLIT_RE = re.compile(r"(?<=[,.!?।\n])\s+")
# Don't flush a clause this short on its own -- e.g. "Well," or "Haan," --
# it sounds abrupt in TTS and saves almost no time. Wait for a bit more text.
EARLY_FLUSH_MIN_CHARS = 18

# ── Fixed offline error strings (never sent to any network TTS) ──
MSG_NO_INTERNET = "Can't connect to the internet."
MSG_NO_SERVER   = "Can't connect to the server."
MSG_TRY_AGAIN   = "Please try again."

# Spoken instead of dead silence when a turn ends with nothing usable to say
# (e.g. the whole reply got trimmed as a truncated fragment). Regular TTS
# voices, not the offline espeak path -- this isn't a connectivity error.
FALLBACK_EN = "Sorry, could you ask that again?"
FALLBACK_HI = "Maaf kijiye, kya aap dobara pooch sakte hain?"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────
#  LATENCY INSTRUMENTATION
# ──────────────────────────────────────────────
# One shared dict per turn: speech_end → stt_complete → llm_start →
# llm_first_token → first_sentence → tts_start → playback_start.
# Reset at the start of every real (non-idle) listening turn.
TURN_TIMINGS: dict = {}


def _reset_timings():
    TURN_TIMINGS.clear()


def _mark(stage: str):
    if stage not in TURN_TIMINGS:   # first write wins (e.g. first token, first sentence)
        TURN_TIMINGS[stage] = time.time()


def print_turn_timings():
    """Prints the requested per-turn latency breakdown, skipping any stage
    that didn't fire this turn (e.g. GPT stages on a reply-cache hit)."""
    t = TURN_TIMINGS
    if "speech_end" not in t:
        return

    def gap(a, b):
        return int((t[b] - t[a]) * 1000) if a in t and b in t else None

    rows = [
        ("Speech End",         0),
        ("STT",                gap("speech_end", "stt_complete")),
        ("LLM First Token",    gap("llm_start", "llm_first_token")),
        ("Sentence Complete",  gap("llm_first_token", "first_sentence")),
        ("TTS Start",          gap("first_sentence", "tts_start")),
        ("Playback Started",   gap("tts_start", "playback_start")),
    ]

    print("\n   ── Latency breakdown ──────────────")
    for label, ms in rows:
        shown = f"{ms} ms" if ms is not None else "n/a (skipped this turn)"
        print(f"   {label:<20} {shown}")
    if "playback_start" in t:
        total = int((t["playback_start"] - t["speech_end"]) * 1000)
        print(f"   {'Total Latency':<20} {total} ms")
    print("   ────────────────────────────────────\n")


# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

STATE_LABEL = {
    State.IDLE:      "😴 IDLE",
    State.LISTENING: "👂 LISTENING",
    State.THINKING:  "🤔 THINKING",
    State.SPEAKING:  "🔊 SPEAKING",
}


def show_state(state: State, note: str = ""):
    """Single always-visible line so the current state is obvious in the terminal."""
    line = f"\n[{STATE_LABEL[state]}]"
    if note:
        line += f" {note}"
    print(line)


# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# Separate history per language so the model never sees cross-language
# context and stays in the right language naturally. Kept as a STABLE
# ordered prefix (system + history) on every call so Sarvam's server-side
# prefix caching can match the repeated prefix.
history: dict = {"en": [], "hi": []}

# Exact-match reply cache: skips the LLM entirely for a repeated question.
reply_cache: dict = {}

# In-memory (voice, text) → path cache in front of the on-disk audio cache.
# The disk cache still does the heavy lifting (survives restarts), this
# just avoids a redundant os.path.exists() stat() call for phrases that
# repeat often within one run (greetings, idle prompts, common replies).
_session_audio_cache: dict = {}

pygame.mixer.init()

# Background workers: one pool for STT segments, one for TTS synthesis.
stt_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stt")
tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")


# ──────────────────────────────────────────────
#  CONNECTIVITY / ERROR HANDLING  (features 5, 6, 7)
# ──────────────────────────────────────────────

def is_internet_available(timeout: float = 2.0) -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def speak_offline(text: str):
    """
    Offline, dependency-free TTS via the local `espeak` binary.
    Used ONLY for error announcements, since it never touches the
    network — unlike the online Sarvam TTS API, which would itself fail if the
    internet or the API is the actual problem.
    """
    print(f"   🔇 (offline) {text}")
    try:
        subprocess.run(
            ["espeak", text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("   ⚠️  espeak is not installed — install it for spoken error messages "
              "(e.g. `sudo apt install espeak`).")


def handle_error(e: Exception, where: str):
    """
    Central error classifier. Order matters:
      1. No internet at all               → "Can't connect to the internet."
      2. Internet is fine but the Sarvam
         API call itself failed           → "Can't connect to the server."
      3. Anything else                    → "Please try again."
    """
    print(f"\n❌ Error in {where}: {type(e).__name__}: {e}")

    if not is_internet_available():
        speak_offline(MSG_NO_INTERNET)
        return

    if isinstance(e, ApiError):
        speak_offline(MSG_NO_SERVER)
        return

    speak_offline(MSG_TRY_AGAIN)


# ──────────────────────────────────────────────
#  AUDIO CACHE  (feature 2 — part of "prompt caching")
# ──────────────────────────────────────────────

def _cache_path(text: str, voice: str, lang: str) -> str:
    key = hashlib.sha256(f"{lang}::{voice}::{text}".encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.mp3")


def _tts_language(lang: str) -> str:
    """Map the bot language to Sarvam Bulbul's BCP-47 language code."""
    return "hi-IN" if lang == "hi" else "en-IN"


def _tts_stream_chunks(text: str, voice: str, lang: str):
    """Yield MP3 chunks from Sarvam Bulbul v3 HTTP streaming TTS."""
    return client.text_to_speech.convert_stream(
        text=text,
        model=TTS_MODEL,
        language_code=_tts_language(lang),
        speaker=voice,
        pace=TTS_PACE,
        temperature=TTS_TEMPERATURE,
        speech_sample_rate=TTS_SAMPLE_RATE,
        output_audio_codec="mp3",
        output_audio_bitrate="128k",
    )


def synthesize(text: str, voice: str, lang: str) -> str:
    """
    Generate Sarvam Bulbul v3 MP3 and cache it on disk.

    Uses Sarvam's HTTP streaming TTS endpoint even for cached synthesis, so
    audio starts arriving as soon as the first chunk is ready.
    """
    session_key = (lang, voice, text)
    cached = _session_audio_cache.get(session_key)
    if cached:
        return cached

    path = _cache_path(text, voice, lang)
    if os.path.exists(path):
        _session_audio_cache[session_key] = path
        return path

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            got_audio = False
            for chunk in _tts_stream_chunks(text, voice, lang):
                if not chunk:
                    continue
                f.write(chunk)
                got_audio = True
        if not got_audio:
            raise RuntimeError("Sarvam Bulbul produced no audio chunks")
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise

    _session_audio_cache[session_key] = path
    return path


def play(path: str):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(50)
    pygame.mixer.music.unload()


# ──────────────────────────────────────────────
#  VOICE SELECTION
# ──────────────────────────────────────────────

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


def _tts_stream_to_mpg123(text: str, voice: str, lang: str, tmp_path: str):
    """
    Stream Sarvam Bulbul v3 MP3 chunks directly into mpg123 while also
    writing the exact same bytes to the persistent audio cache.
    """
    proc = subprocess.Popen(
        ["mpg123", "-q", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    first_chunk = True
    try:
        with open(tmp_path, "wb") as f:
            for data in _tts_stream_chunks(text, voice, lang):
                if not data:
                    continue
                f.write(data)
                try:
                    proc.stdin.write(data)
                    if first_chunk:
                        proc.stdin.flush()
                        _mark("playback_start")
                        first_chunk = False
                except BrokenPipeError:
                    raise RuntimeError("mpg123 stopped before Sarvam TTS finished")
    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError, AttributeError):
            pass
        proc.wait()

    if first_chunk:
        raise RuntimeError("Sarvam Bulbul produced no audio chunks")


def _synth_and_play_streaming(text: str, voice: str, lang: str):
    """
    Used ONLY for the first spoken sentence of a turn (see StreamingSpeaker).
    Starts audio playing as soon as the first chunk of synthesised speech
    exists, instead of waiting for the whole sentence to finish. Falls back
    to the plain synthesize()+play() path if mpg123 isn't installed or
    streaming fails for any other reason — never worse than the original
    behaviour, just not faster.
    """
    _mark("tts_start")

    session_key = (lang, voice, text)
    cached = _session_audio_cache.get(session_key) or (
        _cache_path(text, voice, lang) if os.path.exists(_cache_path(text, voice, lang)) else None
    )
    if cached:
        # Already synthesised in a previous run/turn — nothing to stream.
        _mark("playback_start")
        play(cached)
        _session_audio_cache[session_key] = cached
        return

    path = _cache_path(text, voice, lang)
    tmp_path = path + ".tmp"
    try:
        _tts_stream_to_mpg123(text, voice, lang, tmp_path)
        os.replace(tmp_path, path)
        _session_audio_cache[session_key] = path
    except Exception as e:
        print(f"   ⚠️  streaming TTS playback unavailable ({e}) — falling back to normal playback")
        p = synthesize(text, voice, lang)
        _mark("playback_start")
        play(p)


# ──────────────────────────────────────────────
#  STREAMING SPEAK  — plays sentences as they arrive (feature 1, output half)
# ──────────────────────────────────────────────

class StreamingSpeaker:
    """
    Consumer that plays synthesised sentences in order while the producer
    (LLM stream) is still generating later sentences. Synthesis for
    sentence N+1 happens in the background while sentence N is playing.
    """

    def __init__(self, lang: str):
        self.lang = lang
        self._q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        self._announced = False
        self._sentence_index = 0

    def _consume(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            mode, future = item
            if mode == "sync":
                # Streaming path already played the audio itself inside the
                # worker thread — just block here until it's actually done,
                # so sentence order is still preserved.
                future.result()
            else:
                path = future.result()
                play(path)

    def say(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:                       # feature 4 — never pass empty text
            return
        if not self._announced:
            show_state(State.SPEAKING)
            self._announced = True
        print(f"   💬 {sentence}")
        voice = pick_voice(sentence, self.lang)
        self._sentence_index += 1

        if STREAM_TTS_PLAYBACK and self._sentence_index == 1:
            # Only sentence #1: every later sentence already overlaps its
            # synthesis with the previous sentence's playback below, so
            # streaming only matters for the one sentence with nothing to
            # overlap with. Ordering stays safe because this future does
            # its own playback synchronously — the consumer just waits on it.
            future = tts_executor.submit(_synth_and_play_streaming, sentence, voice, self.lang)
            self._q.put(("sync", future))
        else:
            future = tts_executor.submit(synthesize, sentence, voice, self.lang)
            self._q.put(("play", future))

    def finish(self):
        self._q.put(None)
        self._thread.join()


def speak_blocking(text: str, lang: str = "en"):
    """Simple one-shot speak for short fixed prompts (greeting, idle, wake-ack)."""
    text = text.strip()
    if not text:
        return
    voice = pick_voice(text, lang)
    path = synthesize(text, voice, lang)
    play(path)


# ──────────────────────────────────────────────
#  VAD RECORDING WITH MID-TURN SENTENCE SEGMENTATION  (feature 1, input half)
# ──────────────────────────────────────────────

def calibrate_noise_floor(seconds: float = 1.0) -> float:
    """
    Samples ambient mic noise for `seconds` at startup and sets the module-
    level ENERGY_THRESHOLD just above it. A fixed threshold tuned in a quiet
    room is a real accuracy problem once RoboBot is deployed somewhere loud
    (airport concourse, reception hall) or somewhere very quiet (empty
    classroom): too low → crowd noise falsely triggers "recording", eating
    into the LISTENING window and sometimes clipping the start of the actual
    question; too high → normal speech volume never crosses it and the turn
    silently never starts. Falls back to the existing default if the mic
    can't be read for any reason (never blocks startup).
    """
    global ENERGY_THRESHOLD
    try:
        frames = int(SAMPLE_RATE * seconds)
        sample = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
        sd.wait()
        noise_rms = float(np.sqrt(np.mean(sample ** 2)))
        # Headroom above the measured floor so normal room tone never
        # self-triggers, clamped to a sane range either direction.
        threshold = max(MIN_ENERGY_THRESHOLD, min(MAX_ENERGY_THRESHOLD, noise_rms * 3 + 0.02))
        ENERGY_THRESHOLD = threshold
        print(f"   🎚️  Mic calibrated — noise floor {noise_rms:.4f}, threshold set to {threshold:.4f}")
    except Exception as e:
        print(f"   ⚠️  Mic calibration skipped ({type(e).__name__}: {e}) — using default threshold {ENERGY_THRESHOLD}")
    return ENERGY_THRESHOLD


def transcribe_segment(audio: np.ndarray) -> Tuple[str, str]:
    """Transcribe one audio segment with Sarvam Saaras. Returns (text, lang).

    Sarvam works best with 16 kHz audio, which is already the configured
    SAMPLE_RATE. We use language_code="unknown" so Sarvam can auto-detect
    Hindi/English, then keep the existing script-scan as the final language
    decision for the assistant.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    buf.name = "segment.wav"

    if DEBUG_STT_TIMING:
        audio_secs = len(audio) / SAMPLE_RATE
        _t_encode_done = time.time()
        _t_api_start = time.time()

    result = client.speech_to_text.transcribe(
        file=buf,
        model=STT_MODEL,
        language_code="unknown",
        mode="transcribe",
    )

    if DEBUG_STT_TIMING:
        _t_api_done = time.time()
        request_id = getattr(result, "request_id", None)
        print(
            f"   ⏱️  [STT DEBUG] audio={audio_secs:.2f}s  "
            f"encode={(_t_api_start - _t_encode_done)*1000:.0f}ms  "
            f"api_call={(_t_api_done - _t_api_start)*1000:.0f}ms  "
            f"request_id={request_id}"
        )

    text = (getattr(result, "transcript", "") or "").strip()
    if not text:
        return "", "en"

    # Script scan remains the authoritative language decision, just as in the
    # original design. Devanagari/Urdu -> Hindi; otherwise -> English.
    #
    # NOTE: this is proportion-based, not "any single character". A
    # multilingual ASR model will occasionally transliterate one stray word
    # (a filler sound, a loanword, a proper noun) into Devanagari even when
    # the speaker was clearly speaking English -- treating that one
    # character as proof of Hindi was flipping otherwise-English segments
    # to Hindi and dragging the whole turn's reply language with it.
    lang = _script_scan_lang(text)

    return text, lang


def _script_scan_lang(text: str) -> str:
    """Decide hi/en from script composition, requiring the Devanagari/Urdu
    share of the *lettered* characters to be non-trivial (not just one
    stray character) before calling it Hindi."""
    hi_count = 0
    letter_count = 0
    for ch in text:
        cp = ord(ch)
        is_hi = 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF
        is_latin_letter = ch.isalpha() and cp < 0x0250  # rough Latin-letter range
        if is_hi:
            hi_count += 1
            letter_count += 1
        elif is_latin_letter:
            letter_count += 1
    if letter_count == 0:
        return "en"
    # Require Devanagari/Urdu to be a real fraction of the lettered content,
    # not a single stray transliterated word.
    return "hi" if (hi_count / letter_count) >= 0.3 else "en"


def capture_and_transcribe(timeout: float, track_timing: bool = False) -> Tuple[Optional[str], str]:
    """
    Records one full user turn via VAD. Whenever a short mid-turn pause
    (SENTENCE_PAUSE) is detected, the audio collected so far is sent off
    for transcription in the background immediately — while the user is
    still talking — instead of waiting for the whole turn to end.

    `track_timing=True` resets and records the per-turn latency dict used
    by print_turn_timings() — only set for real listening turns, not idle
    wake-word polling.

    Returns (combined_text_or_None, lang).
    """
    if track_timing:
        _reset_timings()

    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
        blocksize=blocksize, callback=callback,
    )
    stream.start()

    pre_buffer: list = []
    segment_buffer: list = []
    recording = False
    silence_start: Optional[float] = None
    segment_split_done = False
    any_mid_flush = False   # True once ≥1 background segment has been sent for STT
    idle_clock = time.time()
    futures: List = []

    def flush_segment():
        nonlocal segment_buffer
        if not segment_buffer:
            return
        audio = np.concatenate(segment_buffer, axis=0)
        segment_buffer = []
        if len(audio) < SAMPLE_RATE * MIN_SPEECH_SECS:   # reject locally -- no round trip
            return
        futures.append(stt_executor.submit(transcribe_segment, audio))
        print("   ✂️  segment captured → transcribing in background...")

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    stream.stop(); stream.close()
                    return None, "en"
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock = time.time()
                silence_start = None
                segment_split_done = False
                if not recording:
                    recording = True
                    segment_buffer = list(pre_buffer)
                segment_buffer.append(chunk)

            elif recording:
                segment_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                    continue
                elapsed = time.time() - silence_start
                required_silence = SILENCE_AFTER_SPEECH_FOLLOWUP if any_mid_flush else SILENCE_AFTER_SPEECH
                if elapsed >= required_silence:
                    if track_timing:
                        _mark("speech_end")
                    break  # end of the whole turn
                if elapsed >= SENTENCE_PAUSE and not segment_split_done:
                    segment_split_done = True
                    any_mid_flush = True
                    flush_segment()

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    stream.stop(); stream.close()
                    return None, "en"
    finally:
        stream.stop()
        stream.close()

    flush_segment()  # final trailing segment

    if not futures:
        return None, "en"

    pieces = []
    lang = "en"
    for fut in futures:
        try:
            text, seg_lang = fut.result()
        except Exception:
            continue
        if text:
            pieces.append(text)
            lang = seg_lang  # last non-empty segment's script-scan wins

    if track_timing:
        _mark("stt_complete")

    combined = " ".join(p for p in pieces if p.strip()).strip()

    # Re-run the script scan over the FULL combined text so language never
    # flips mid-turn just because one short segment mis-detected. This uses
    # the same proportion-based decision as transcribe_segment() -- a single
    # stray Devanagari character from one mis-transcribed segment should not
    # be able to override an otherwise clearly-English turn.
    if combined:
        lang = _script_scan_lang(combined)

    if not combined or len(combined) < 1:
        return None, lang

    return combined, lang


# ──────────────────────────────────────────────
#  WAKE WORD
# ──────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ──────────────────────────────────────────────
#  AI REPLY — streamed, sentence-by-sentence, with reply caching
# ──────────────────────────────────────────────

def stream_ai_reply_and_speak(user_text: str, lang: str) -> str:
    """
    Streams the LLM reply and speaks it sentence-by-sentence as it's
    generated. Returns the full reply text (for history bookkeeping,
    which already happened inline).
    """
    system = SYSTEM_HI if lang == "hi" else SYSTEM_EN
    lang_history = history[lang]

    cache_key = (lang, user_text.strip().lower())
    speaker = StreamingSpeaker(lang)

    # ── Prompt cache hit: skip the LLM call entirely ─────────────
    if cache_key in reply_cache:
        print("   💾 cache hit — skipping LLM call")
        cached_reply = reply_cache[cache_key]
        lang_history.append({"role": "user", "content": user_text})
        lang_history.append({"role": "assistant", "content": cached_reply})
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(lang_history) > max_msgs:
            del lang_history[:len(lang_history) - max_msgs]
        for sentence in SENTENCE_SPLIT_RE.split(cached_reply):
            speaker.say(sentence)
        speaker.finish()
        return cached_reply

    # ── Stable prefix (system + history) so Sarvam's server-side
    #    prefix caching can match repeated prefixes across turns ──
    # NOTE: user_text is NOT appended to lang_history here. If this turn
    # ends up producing nothing usable (error, or the whole reply gets
    # trimmed as a truncated fragment), a dangling user turn with no
    # assistant reply would sit in history forever and corrupt every
    # future prompt built from it. It's only appended below once we know
    # the turn actually succeeded.
    messages = [
        {"role": "system", "content": system},
        *lang_history,
        {"role": "user", "content": user_text},
    ]

    _mark("llm_start")
    stream = client.chat.completions(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        reasoning_effort=None,
        stream=True,
    )

    buffer = ""
    full_reply = ""
    first_chunk_flushed = False   # switches EARLY_SPLIT_RE -> SENTENCE_SPLIT_RE after chunk #1
    finish_reason = None

    for chunk in stream:
        # Sarvam's final usage chunk can have an empty `choices` array.
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason or finish_reason
        delta = (choice.delta.content or "")
        if not delta:                     # feature 4 — skip empty deltas
            continue
        _mark("llm_first_token")
        buffer += delta
        full_reply += delta

        if not first_chunk_flushed:
            # Only for the very first spoken chunk: split on a clause-level
            # comma too, so playback can start on the first clause instead
            # of waiting for a whole sentence — this is what keeps replies
            # inside budget even when the model's first sentence runs long.
            parts = EARLY_SPLIT_RE.split(buffer)
            if len(parts) > 1 and len(parts[0]) >= EARLY_FLUSH_MIN_CHARS:
                _mark("first_sentence")
                speaker.say(parts[0])
                first_chunk_flushed = True
                buffer = " ".join(parts[1:])
            continue

        parts = SENTENCE_SPLIT_RE.split(buffer)
        if len(parts) > 1:
            _mark("first_sentence")   # no-op after chunk #1 (first write wins)
            for sentence in parts[:-1]:
                speaker.say(sentence)
            buffer = parts[-1]

    if buffer.strip():
        if finish_reason == "length":
            # Hard-cut by MAX_TOKENS mid-sentence/mid-word — don't speak a
            # severed fragment. Trim back to the last complete sentence; if
            # there isn't even one complete sentence in this trailing chunk,
            # drop it entirely rather than read out a broken half-thought.
            # Only the trailing fragment is dropped — sentences already
            # spoken earlier in this turn are untouched.
            original_tail = buffer
            complete = SENTENCE_SPLIT_RE.split(buffer)
            buffer = " ".join(s.strip() for s in complete[:-1] if s.strip()) if len(complete) > 1 else ""
            full_reply = full_reply[: len(full_reply) - len(original_tail)] + buffer
        if buffer.strip():
            _mark("first_sentence")   # covers single-sentence / single-clause replies too
            speaker.say(buffer)

    full_reply = full_reply.strip()
    if not full_reply:
        # Nothing usable came out of this turn (empty stream, or the whole
        # reply was a single truncated fragment with no complete sentence).
        # Speak a fallback instead of dead silence, and leave history alone
        # — there's nothing real to build the next prompt on.
        speaker.say(FALLBACK_HI if lang == "hi" else FALLBACK_EN)

    speaker.finish()

    if full_reply:                        # feature 4 — don't cache/store empties
        lang_history.append({"role": "user", "content": user_text})
        lang_history.append({"role": "assistant", "content": full_reply})
        reply_cache[cache_key] = full_reply
        # Cap history so prompt size — and therefore LLM First Token latency
        # — stays flat over a multi-hour deployment instead of creeping up
        # turn by turn. Keep the most recent MAX_HISTORY_TURNS user+assistant
        # pairs only.
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(lang_history) > max_msgs:
            del lang_history[:len(lang_history) - max_msgs]

    return full_reply


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 56)
    print("  🤖 RoboBot — streaming speech-to-speech assistant")
    print("=" * 56)
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' to wake up")
    print("    🤔 THINKING   — waiting on the first tokens")
    print("    🔊 SPEAKING   — plays each sentence as it's ready")
    print("  Ctrl+C to quit")
    print("=" * 56 + "\n")


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print_banner()

    state = State.LISTENING
    lang  = "hi"

    calibrate_noise_floor()

    try:speak_blocking('Hi , I am aayro . Aapka swagat hai. Aap apna sawaal poochhiye. ', lang="hi")
    except Exception as e:
        handle_error(e, "opening greeting")

    try:
        while True:

            # ════════════════════════ IDLE ════════════════════════
            if state == State.IDLE:
                show_state(state, "— say 'Hello' to activate...")
                try:
                    combined, _ = capture_and_transcribe(timeout=IDLE_POLL_TIMEOUT)
                except Exception as e:
                    handle_error(e, "idle listening")
                    continue

                if not combined:
                    continue

                print(f"   Heard: {combined!r}")
                if is_wake_word(combined):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    try:
                        speak_blocking("Haan, mein sun rahi hoon. Aap apna sawaal poochhiye.", lang="hi")
                    except Exception as e:
                        handle_error(e, "wake acknowledgement")
                else:
                    print("   Not a wake word — staying idle.")
                continue

            # ═════════════════════ LISTENING ═══════════════════════
            if state == State.LISTENING:
                show_state(state, f"— silence for {int(IDLE_TIMEOUT)}s → idle")
                try:
                    user_text, lang = capture_and_transcribe(timeout=IDLE_TIMEOUT, track_timing=True)
                except Exception as e:
                    handle_error(e, "listening / transcription")
                    continue

                if user_text is None:
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    try:
                        speak_blocking(
                            "Mein abhi idle mode mein ja rahi hoon. Jab zaroorat ho, 'Hello' kahiye.",
                            lang="hi",
                        )
                    except Exception as e:
                        handle_error(e, "idle announcement")
                    continue

                user_text = user_text.strip()
                if not user_text:             # feature 4 — never forward empty text
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")
                state = State.THINKING
                continue

            # ═════════════════════ THINKING ════════════════════════
            if state == State.THINKING:
                show_state(state)
                try:
                    stream_ai_reply_and_speak(user_text, lang)
                except Exception as e:
                    handle_error(e, "LLM reply / speech synthesis")
                    state = State.LISTENING
                    continue
                if PRINT_LATENCY_TIMINGS:
                    print_turn_timings()
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        stt_executor.shutdown(wait=False, cancel_futures=True)
        tts_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
