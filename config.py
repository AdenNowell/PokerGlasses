import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class SystemConfig:
    # Mode: "CHESS" or "POKER"
    GAME_MODE: str = os.getenv("GAME_MODE", "POKER").upper()
    
    # Capture & Gating Parameters
    CAPTURE_FPS: int = 2
    JEV_CONFIDENCE_THRESHOLD: float = 0.30
    MOTION_SETTLE_FLOOR: float = 0.015  # Min diff to consider change
    MOTION_SETTLE_CEILING: float = 0.08 # Max diff to avoid capturing moving hands
    
    # Screen Capture Region (WhatsApp Video Window on macOS)
    WINDOW_TOP: int = int(os.getenv("WINDOW_TOP", 100))
    WINDOW_LEFT: int = int(os.getenv("WINDOW_LEFT", 100))
    WINDOW_WIDTH: int = int(os.getenv("WINDOW_WIDTH", 800))
    WINDOW_HEIGHT: int = int(os.getenv("WINDOW_HEIGHT", 600))
    
    # API Keys & Endpoints
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "claude-opus-4-20250514") # Or custom Opus 5.5 endpoint
    
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB") # Adam
    VIRTUAL_AUDIO_DEVICE: str = os.getenv("VIRTUAL_AUDIO_DEVICE", "BlackHole 2ch")
    
    # Engines
    STOCKFISH_PATH: str = os.getenv("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")
    MONTE_CARLO_ITERATIONS: int = 10000
    
    # Telemetry (Supabase)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    TELEMETRY_BATCH_SIZE: int = 10

CONFIG = SystemConfig()
