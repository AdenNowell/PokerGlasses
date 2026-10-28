import random
from typing import List, Dict, Any, Callable, Optional
from treys import Card, Evaluator, Deck
from config import CONFIG


class PokerEngine:
    """
    Fast Monte Carlo poker equity calculator using bitwise LUTs (treys),
    with extensible hooks for game-theoretic strategy vector mapping.
    """
    def __init__(self, iterations: int = CONFIG.MONTE_CARLO_ITERATIONS):
        self.evaluator = Evaluator()
        self.iterations = iterations
        self.strategy_hook: Optional[Callable[[float, int, int], Dict[str, float]]] = None

    def register_strategy_hook(self, hook_fn: Callable[[float, int, int], Dict[str, float]]) -> None:
        """
        Registers a custom game-theoretic vector mapper.
        Signature: fn(equity: float, street_cards: int, num_opponents: int) -> Dict[action, weight]
        """
        self.strategy_hook = hook_fn

    def evaluate_hand(
        self,
        hole_strs: List[str],
        board_strs: List[str],
        num_opponents: int = 1
    ) -> Dict[str, Any]:
        if len(hole_strs) != 2:
            return {"valid": False, "speech": "Waiting for hole cards."}

        try:
            hole_cards = [Card.new(c) for c in hole_strs]
            board_cards = [Card.new(c) for c in board_strs]
        except Exception:
            return {"valid": False, "speech": "Card parse error."}

        known_cards = set(hole_cards + board_cards)
        full_deck = [c for c in Deck.GetFullDeck() if c not in known_cards]
        cards_to_deal = 5 - len(board_cards)
        needed_per_sim = cards_to_deal + (2 * max(1, num_opponents))

        wins = 0.0
        for _ in range(self.iterations):
            sampled = random.sample(full_deck, needed_per_sim)
            sim_board = board_cards + sampled[:cards_to_deal]
            hero_rank = self.evaluator.evaluate(sim_board, hole_cards)

            opp_ranks = []
            idx = cards_to_deal
            for _ in range(max(1, num_opponents)):
                opp_hand = sampled[idx:idx + 2]
                opp_ranks.append(self.evaluator.evaluate(sim_board, opp_hand))
                idx += 2

            best_opp_rank = min(opp_ranks)  # Lower rank integer = stronger hand in treys
            if hero_rank < best_opp_rank:
                wins += 1.0
            elif hero_rank == best_opp_rank:
                wins += 0.5

        equity = (wins / self.iterations) * 100.0

        # Execute asymmetric / GTO strategy vector hook if registered, else use default mixed vector
        strategy_vector = (
            self.strategy_hook(equity, len(board_cards), num_opponents)
            if self.strategy_hook
            else self._default_asymmetric_vector(equity, len(board_cards), num_opponents)
        )
        recommended_action = max(strategy_vector, key=strategy_vector.get)

        speech = f"{round(equity)} percent equity. {recommended_action}."
        return {
            "valid": True,
            "equity_pct": round(equity, 2),
            "num_opponents": num_opponents,
            "strategy_vector": strategy_vector,
            "recommended_action": recommended_action,
            "speech": speech,
        }

    @staticmethod
    def _default_asymmetric_vector(equity: float, board_len: int, num_opponents: int) -> Dict[str, float]:
        """
        Maps raw equity vs random range baseline (1 / (N+1)) into a normalized
        mixed-strategy probability distribution across [Fold, Check/Call, Value Bet, Polar Jam].
        """
        fair_share = 100.0 / (num_opponents + 1)
        edge = equity - fair_share

        if edge < -10.0:
            return {"Fold": 0.85, "Check-Call": 0.10, "Aggro-Bluff": 0.05}
        elif -10.0 <= edge < 8.0:
            return {"Fold": 0.15, "Check-Call": 0.70, "Small-Probe": 0.15}
        elif 8.0 <= edge < 25.0:
            return {"Check-Call": 0.25, "Standard-Raise": 0.65, "Overbet": 0.10}
        else:
            return {"Standard-Raise": 0.30, "Heavy-Value": 0.70}
