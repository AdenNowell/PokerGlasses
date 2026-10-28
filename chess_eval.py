from typing import Dict, Any
from stockfish import Stockfish
from config import CONFIG


class ChessEngine:
    """
    Interfaces with a local Stockfish binary for sub-second FEN evaluation.
    """
    FILE_NAMES = {
        "a": "Alpha", "b": "Bravo", "c": "Charlie", "d": "Delta",
        "e": "Echo", "f": "Foxtrot", "g": "Golf", "h": "Hotel"
    }

    def __init__(self, binary_path: str = CONFIG.STOCKFISH_PATH):
        self.engine = Stockfish(
            path=binary_path,
            depth=15,
            parameters={"Threads": 4, "Hash": 512, "Minimum Thinking Time": 20}
        )

    def evaluate_fen(self, fen: str) -> Dict[str, Any]:
        if not self.engine.is_fen_valid(fen):
            return {"valid": False, "speech": "Invalid board state."}

        self.engine.set_fen_position(fen)
        best_move = self.engine.get_best_move_time(250)  # 250ms calculation cap
        evaluation = self.engine.get_evaluation()

        if not best_move:
            return {"valid": False, "speech": "No legal moves found."}

        spoken_move = self._format_move_for_speech(best_move, evaluation)
        return {
            "valid": True,
            "best_move": best_move,
            "eval_type": evaluation.get("type"),
            "eval_value": evaluation.get("value"),
            "speech": spoken_move,
        }

    def _format_move_for_speech(self, uci_move: str, evaluation: Dict[str, Any]) -> str:
        src, dst = uci_move[:2], uci_move[2:4]
        eval_str = ""
        if evaluation["type"] == "cp":
            score = evaluation["value"] / 100.0
            sign = "plus" if score > 0 else "minus" if score < 0 else "even"
            eval_str = f", {sign} {abs(score):.1f}"
        elif evaluation["type"] == "mate":
            eval_str = f", mate in {abs(evaluation['value'])}"

        return f"{src.upper()} to {dst.upper()}{eval_str}."
