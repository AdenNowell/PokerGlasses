# meta-game-sense- Finally Added

## Implementation status

This repository assembles the supplied prototype. Python syntax has been checked;
dependencies, external APIs, screen capture, engines, and audio routing have not
been integration-tested. Use for permitted practice or analysis.

- The Jev wrapper currently uses an image-histogram heuristic, not a Jev model.
- Capture uses a configured screen rectangle; it does not discover the active window.
- Set `VISION_MODEL` in `.env` to a vision-capable model available to your account.
- A single image cannot reliably establish chess turn, castling rights, or en passant
  history. Verify extracted FEN and card values before relying on results.
- Poker results assume random opponent hands. The default strategy weights are
  illustrative heuristics, not a GTO solver. Multiway ties currently receive a
  fixed half-share, so multiway equity requires correction before use.
- Speech playback and telemetry flushing run synchronously. Failed telemetry
  batches are discarded. Reported latency excludes speech playback and logging.
- No API-call reduction or runtime benchmarks have been measured.
- The audio output falls back to the system default if the virtual device is absent.
- WhatsApp/glasses compatibility and UI steps depend on the installed versions.

The original pipeline files are preserved as supplied. API keys and engine binaries
are not included. No GitHub repository has been created and nothing has been pushed.


A low-latency wearable AR game-state pipeline that bridges live POV video from **Meta Ray-Ban Smart Glasses** (via WhatsApp macOS desktop) to local chess and poker solvers, routing synthesized voice advisories directly back into the glasses' open-ear speakers.

---

## System Architecture

Continuous vision-LLM polling on live video streams is prohibitively expensive and introduces severe queuing latency. `meta-game-sense` solves this using a **two-stage local classifier gate** in front of the frontier vision model:

```text
[Meta Ray-Bans Camera] 
       │ (WhatsApp Video Call)
       ▼
[macOS Window Capture (mss)] ──► [Kinematic Settle + Jev Classifier Gate (Threshold ≥ 0.30)]
                                        │
                    ┌───────────────────┴───────────────────┐
              (State Unchanged)                       (State Changed)
                    │                                       │
              [Drop Frame]                                  ▼
                                              [Opus Vision State Extractor]
                                               ├── Chess: Validated FEN
                                               └── Poker: Hole & Board Cards
                                                            │
                                                            ▼
                                               [Local Evaluation Engines]
                                               ├── Stockfish 16 (UCI C++ Binary)
                                               └── Bitwise LUT Monte Carlo + Strategy Vector
                                                            │
                                                            ▼
                                               [ElevenLabs Flash v2.5 TTS]
                                                            │
                                       ┌────────────────────┴────────────────────┐
                                       ▼                                         ▼
                        [BlackHole 2ch Virtual Mic]                  [Batched Supabase Logger]
                        (Loops back into WhatsApp Call)              (Buffered Telemetry & Latency)
```

---

## Prototype Components

1. **Kinematic Settle + Jev Pre-Check (`vision_gate.py`):** 
   Detects when a dealer's hand or opponent's arm enters the frame, waits for physical motion to settle, and runs a local Jev state-transition classifier (`confidence >= 0.30`). Intended to reduce external vision API calls; savings are unmeasured.
2. **Vision Extraction (`extractor.py`):**
   Prompts Opus with strict zero-temperature Pydantic schemas to extract Forsyth-Edwards Notation (FEN) or 2-character poker card strings (`Ah`, `Kd`, `2s`).
3. **Local Solvers (`eval_engines/`):**
   * **Chess:** Interfaces directly with a local Stockfish binary (`depth=15`, 250ms budget).
   * **Poker:** Runs 10,000-iteration bitwise Lookup Table (LUT) Monte Carlo simulations via `treys` (runtime is unmeasured), passing raw equity into an extensible **Strategy Vector Hook** for asymmetric/GTO action mixing.
4. **Zero-Hardware Audio Loopback (`comms.py`):**
   Streams ElevenLabs `eleven_flash_v2_5` audio directly into a macOS CoreAudio virtual loopback driver (`BlackHole 2ch`), acting as the microphone input for the active WhatsApp call.
5. **Batched Telemetry (`comms.py`):**
   Buffers frame confidence scores, parsed states, and end-to-end loop latencies in memory and flushes them in batch inserts to Supabase on the main inference thread; batch writes can block it.

---

## macOS Setup & Virtual Audio Routing

### 1. Install System Binaries
```bash
brew install stockfish
brew install --cask blackhole-2ch
```

### 2. Configure WhatsApp Audio Loopback
1. Open **Audio MIDI Setup** on macOS (`Cmd + Space` -> `Audio MIDI Setup`).
2. Verify **BlackHole 2ch** appears as an audio device (44.1kHz or 48kHz).
3. Open **WhatsApp Desktop**, start a video call with the secondary account linked to your Meta Ray-Bans, and double-click the glasses' capture button to switch the video feed to the glasses POV.
4. In WhatsApp Settings -> **Video & Voice**, set **Microphone** to `BlackHole 2ch`.

### 3. Supabase Schema
Run the following SQL in your Supabase SQL Editor to initialize the telemetry table:
```sql
create table public.meta_game_sense_logs (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    timestamp_epoch double precision not null,
    game_mode text not null,
    jev_confidence double precision not null,
    latency_ms double precision not null,
    parsed_state jsonb not null,
    engine_output jsonb not null
);
```

### 4. Environment & Execution
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Populate ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, SUPABASE_URL, SUPABASE_KEY
python main.py
```
