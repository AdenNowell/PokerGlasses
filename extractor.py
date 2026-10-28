import json
from typing import List, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic
from config import CONFIG


class ChessState(BaseModel):
    valid_board_visible: bool = Field(description="True if a chess board is clearly visible")
    fen: Optional[str] = Field(default=None, description="Full Forsyth-Edwards Notation string")
    active_color: str = Field(default="w", description="'w' or 'b'")


class PokerState(BaseModel):
    valid_table_visible: bool = Field(description="True if poker cards are clearly visible")
    hole_cards: List[str] = Field(default_factory=list, description="Hero's hole cards, e.g., ['Ah', 'Kd']")
    community_cards: List[str] = Field(default_factory=list, description="Board cards, e.g., ['2s', '4c', '9h']")
    num_active_opponents: int = Field(default=1, description="Estimated number of active opponents in hand")


class StateExtractor:
    """
    Interfaces with the frontier vision model to convert gated frames
    into deterministic game-state objects.
    """
    def __init__(self):
        self.client = Anthropic(api_key=CONFIG.ANTHROPIC_API_KEY)
        self.model = CONFIG.VISION_MODEL

    def extract_chess_state(self, b64_image: str) -> ChessState:
        system_prompt = (
            "You are an expert chess vision parser analyzing POV smart-glasses video. "
            "Extract the exact board position into a valid FEN string from the wearer's perspective. "
            "Respond ONLY with raw JSON matching: "
            '{"valid_board_visible": bool, "fen": "string or null", "active_color": "w" or "b"}'
        )
        raw_json = self._call_vision_api(b64_image, system_prompt)
        try:
            data = json.loads(raw_json)
            return ChessState(**data)
        except Exception:
            return ChessState(valid_board_visible=False)

    def extract_poker_state(self, b64_image: str) -> PokerState:
        system_prompt = (
            "You are a precision poker vision parser analyzing POV smart-glasses video. "
            "Identify the wearer's hole cards (closest to bottom/hands) and community board cards. "
            "Use 2-character notation: Rank (2-9, T, J, Q, K, A) + Suit (s, h, d, c). Example: 'Ah', 'Td', '2c'. "
            "Respond ONLY with raw JSON matching: "
            '{"valid_table_visible": bool, "hole_cards": ["Ah", "Kd"], '
            '"community_cards": ["2s", "4c", "9h"], "num_active_opponents": int}'
        )
        raw_json = self._call_vision_api(b64_image, system_prompt)
        try:
            data = json.loads(raw_json)
            return PokerState(**data)
        except Exception:
            return PokerState(valid_table_visible=False)

    def _call_vision_api(self, b64_image: str, system_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract the current state JSON.",
                        },
                    ],
                }
            ],
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return text
