import time
from config import CONFIG
from vision_gate import VisionGate
from extractor import StateExtractor
from eval_engines import ChessEngine, PokerEngine
from comms import CommsAndTelemetry


def run_pipeline():
    print(f"[*] Initializing meta-game-sense in [{CONFIG.GAME_MODE}] mode...")
    gate = VisionGate()
    extractor = StateExtractor()
    comms = CommsAndTelemetry()

    chess_engine = ChessEngine() if CONFIG.GAME_MODE == "CHESS" else None
    poker_engine = PokerEngine() if CONFIG.GAME_MODE == "POKER" else None

    last_state_signature = None
    frame_interval = 1.0 / CONFIG.CAPTURE_FPS

    try:
        while True:
            loop_start = time.perf_counter()
            triggered, b64_frame, jev_conf = gate.poll_for_state_change()

            if not triggered or b64_frame is None:
                time.sleep(frame_interval)
                continue

            print(f"\n[GATE TRIGGERED] Jev Confidence: {jev_conf:.2f} | Extracting state...")

            if CONFIG.GAME_MODE == "CHESS":
                state = extractor.extract_chess_state(b64_frame)
                if not state.valid_board_visible or not state.fen:
                    continue

                # Deduplicate identical board states
                board_sig = state.fen.split(" ")[0]
                if board_sig == last_state_signature:
                    continue
                last_state_signature = board_sig

                result = chess_engine.evaluate_fen(state.fen)
                parsed_dict = state.model_dump()

            else:  # POKER
                state = extractor.extract_poker_state(b64_frame)
                if not state.valid_table_visible or len(state.hole_cards) != 2:
                    continue

                state_sig = f"{sorted(state.hole_cards)}_{sorted(state.community_cards)}"
                if state_sig == last_state_signature:
                    continue
                last_state_signature = state_sig

                result = poker_engine.evaluate_hand(
                    state.hole_cards, state.community_cards, state.num_active_opponents
                )
                parsed_dict = state.model_dump()

            latency_ms = (time.perf_counter() - loop_start) * 1000.0
            print(f"[SOLVED in {latency_ms:.0f}ms] -> {result}")

            if result.get("valid"):
                comms.speak_to_glasses(result["speech"])
                comms.log_loop_event(
                    game_mode=CONFIG.GAME_MODE,
                    jev_confidence=jev_conf,
                    latency_ms=latency_ms,
                    parsed_state=parsed_dict,
                    engine_output=result,
                )

            time.sleep(frame_interval)

    except KeyboardInterrupt:
        print("\n[*] Flushing remaining telemetry to Supabase and shutting down...")
        comms.flush_telemetry()


if __name__ == "__main__":
    run_pipeline()
