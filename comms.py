import io
import time
from typing import Dict, Any, List, Optional
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs
from supabase import create_client, Client
from config import CONFIG


class CommsAndTelemetry:
    """
    Manages ElevenLabs TTS routing to macOS BlackHole virtual audio
    and asynchronous batch logging to Supabase.
    """
    def __init__(self):
        self.tts_client = ElevenLabs(api_key=CONFIG.ELEVENLABS_API_KEY) if CONFIG.ELEVENLABS_API_KEY else None
        self.output_device_id = self._resolve_audio_device(CONFIG.VIRTUAL_AUDIO_DEVICE)
        
        self.supabase: Optional[Client] = None
        if CONFIG.SUPABASE_URL and CONFIG.SUPABASE_KEY:
            self.supabase = create_client(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY)
            
        self.telemetry_buffer: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_audio_device(target_name: str) -> Optional[int]:
        """Finds the CoreAudio index for BlackHole 2ch on macOS."""
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if target_name.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
                return idx
        print(f"[WARN] Virtual audio device '{target_name}' not found. Falling back to system default.")
        return None

    def speak_to_glasses(self, text: str) -> None:
        """
        Synthesizes speech via ElevenLabs Flash model and plays it directly
        into the virtual microphone interface.
        """
        print(f"[AUDIO OUT] -> {text}")
        if not self.tts_client:
            return

        audio_stream = self.tts_client.text_to_speech.convert(
            voice_id=CONFIG.ELEVENLABS_VOICE_ID,
            model_id="eleven_flash_v2_5",
            text=text,
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(chunk for chunk in audio_stream)
        data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        sd.play(data, samplerate=samplerate, device=self.output_device_id)
        sd.wait()

    def log_loop_event(
        self,
        game_mode: str,
        jev_confidence: float,
        latency_ms: float,
        parsed_state: Dict[str, Any],
        engine_output: Dict[str, Any]
    ) -> None:
        """
        Queues telemetry events and flushes in batches to minimize I/O overhead.
        """
        payload = {
            "timestamp_epoch": time.time(),
            "game_mode": game_mode,
            "jev_confidence": jev_confidence,
            "latency_ms": round(latency_ms, 2),
            "parsed_state": parsed_state,
            "engine_output": engine_output,
        }
        self.telemetry_buffer.append(payload)

        if len(self.telemetry_buffer) >= CONFIG.TELEMETRY_BATCH_SIZE:
            self.flush_telemetry()

    def flush_telemetry(self) -> None:
        if not self.supabase or not self.telemetry_buffer:
            self.telemetry_buffer.clear()
            return

        batch = self.telemetry_buffer[:]
        self.telemetry_buffer.clear()
        try:
            self.supabase.table("meta_game_sense_logs").insert(batch).execute()
        except Exception as exc:
            print(f"[TELEMETRY WARN] Batch insert failed: {exc}")
