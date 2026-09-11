"""
Chess Puzzle Difficulty Classifier

Classifies chess puzzles into difficulty categories:
- Very Easy (< 800)
- Easy (800-1200)
- Medium (1200-1800)
- Hard (1800-2400)
- Extreme (2400+)
"""

import chess
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict
from feature_extractor import FeatureExtractor


class PuzzleClassifier:
    """Fast CPU-based puzzle difficulty classifier."""
    
    # Difficulty thresholds based on Lichess puzzle ratings
    DIFFICULTY_THRESHOLDS = {
        'Very Easy': (0, 800),
        'Easy': (800, 1200),
        'Medium': (1200, 1800),
        'Hard': (1800, 2400),
        'Extreme': (2400, 5000)
    }
    
    def __init__(self, model_path: str = "models/puzzle_classifier_model.pkl"):
        """
        Initialize classifier with trained model.
        
        Args:
            model_path: Path to pickled model file
        """
        configured_model_path = os.getenv("PUZZLE_CLASSIFIER_MODEL_PATH", model_path)
        resolved_model_path = Path(configured_model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = Path(__file__).resolve().parent / resolved_model_path

        print(f"Loading model from {resolved_model_path}...")
        with open(resolved_model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.feature_extractor = FeatureExtractor()
        print("Classifier ready!")
    
    def _get_final_position_fen(self, start_fen: str, moves_uci: str) -> str:
        """Apply solution moves to get final position FEN."""
        try:
            board = chess.Board(start_fen)
            moves = moves_uci.strip().split()
            
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            
            return board.fen()
        except:
            return start_fen
    
    def _extract_solution_features(self, board: chess.Board, moves_uci: str) -> np.ndarray:
        """Extract features from puzzle solution."""
        moves = moves_uci.strip().split()
        
        num_moves = len(moves)
        num_captures = 0
        num_checks = 0
        num_queen_moves = 0
        num_forced = 0
        material_lost = 0
        
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        try:
            temp_board = board.copy()
            
            for move_uci in moves:
                legal_moves = list(temp_board.legal_moves)
                if len(legal_moves) == 1:
                    num_forced += 1
                
                move = chess.Move.from_uci(move_uci)
                
                if temp_board.is_capture(move):
                    num_captures += 1
                    captured_piece = temp_board.piece_at(move.to_square)
                    if captured_piece:
                        material_lost -= piece_values.get(captured_piece.piece_type, 0)
                
                piece = temp_board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.QUEEN:
                    num_queen_moves += 1
                
                temp_board.push(move)
                
                if temp_board.is_check():
                    num_checks += 1
            
            sacrifice_score = max(0, -material_lost)
            
        except:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        return np.array([
            num_moves,
            num_captures,
            num_checks,
            num_queen_moves,
            num_forced,
            sacrifice_score
        ], dtype=np.float32)
    
    def predict_rating(self, fen: str, solution_moves: str) -> float:
        """
        Predict puzzle rating.
        
        Args:
            fen: Starting position in FEN notation
            solution_moves: Solution in UCI format (e.g., "e2e4 e7e5")
        
        Returns:
            Predicted rating (float)
        """
        # Extract question features
        question_features = self.feature_extractor.extract_features(fen)
        
        # Extract answer features
        final_fen = self._get_final_position_fen(fen, solution_moves)
        answer_features = self.feature_extractor.extract_features(final_fen)
        
        # Extract solution features
        board = chess.Board(fen)
        solution_features = self._extract_solution_features(board, solution_moves)
        
        # Combine all features
        features = np.concatenate([
            question_features,
            answer_features,
            solution_features
        ]).reshape(1, -1)
        
        # Predict
        rating = self.model.predict(features)[0]
        return rating
    
    def classify(self, fen: str, solution_moves: str = None) -> Tuple[str, float]:
        """
        Classify puzzle difficulty.
        
        Args:
            fen: Starting position in FEN notation
            solution_moves: Solution in UCI format (optional, but recommended for accuracy)
        
        Returns:
            Tuple of (difficulty_label, predicted_rating)
            
        Note:
            If solution_moves is not provided, uses position-only heuristics.
            This is significantly less accurate (~50% less predictive).
        """
        if solution_moves is None:
            # Fallback: Use only question position features (very rough estimate)
            rating = self._predict_rating_position_only(fen)
        else:
            rating = self.predict_rating(fen, solution_moves)
        
        # Determine difficulty category
        for category, (low, high) in self.DIFFICULTY_THRESHOLDS.items():
            if low <= rating < high:
                return category, rating
        
        return 'Extreme', rating
    
    def _predict_rating_position_only(self, fen: str) -> float:
        """
        Fallback: Predict rating using only starting position.
        WARNING: Much less accurate than with solution moves!
        """
        import warnings
        warnings.warn(
            "Predicting without solution moves is significantly less accurate. "
            "Expected error: ~400-450 rating points vs 285 with moves.",
            UserWarning
        )
        
        # Extract only question features, pad with zeros for answer+solution
        question_features = self.feature_extractor.extract_features(fen)
        # Pad with zeros for missing answer (24) and solution (6) features
        features = np.concatenate([
            question_features,
            np.zeros(24, dtype=np.float32),  # Missing answer features
            np.zeros(6, dtype=np.float32)    # Missing solution features
        ]).reshape(1, -1)
        
        rating = self.model.predict(features)[0]
        return rating
    
    def classify_batch(self, puzzles: list) -> list:
        """
        Classify multiple puzzles efficiently.
        
        Args:
            puzzles: List of (fen, solution_moves) tuples
        
        Returns:
            List of (difficulty_label, predicted_rating) tuples
        """
        results = []
        for fen, moves in puzzles:
            category, rating = self.classify(fen, moves)
            results.append((category, rating))
        return results
    
    def get_difficulty_stats(self) -> Dict[str, Tuple[int, int]]:
        """Get difficulty category definitions."""
        return self.DIFFICULTY_THRESHOLDS.copy()


def main():
    """Example usage of the classifier."""
    import sys
    
    # Example puzzles
    examples = [
        # Very Easy: Simple mate in 1
        ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - -", "e1e8"),
        
        # Easy: Basic tactic
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -", "f3e5 c6e5 c4f7"),
        
        # Medium: Multi-move combination
        ("r1b1kb1r/pppp1ppp/2n2q2/4p3/2B1n3/3P1N2/PPP2PPP/RNBQK2R w KQkq -", "c4f7 e8f7 d1d5 f6d6 d5e4"),
        
        # Hard: Complex sacrifice
        ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - -", 
         "c4f7 f8f7 c3d5 d8d7 d5f6 g7f6"),
    ]
    
    classifier = PuzzleClassifier()
    
    print("\n" + "="*60)
    print("Puzzle Difficulty Classification Examples")
    print("="*60 + "\n")
    
    for i, (fen, moves) in enumerate(examples, 1):
        category, rating = classifier.classify(fen, moves)
        print(f"Puzzle {i}:")
        print(f"  FEN: {fen[:50]}...")
        print(f"  Difficulty: {category}")
        print(f"  Predicted Rating: {rating:.0f}")
        print()
    
    # Show category definitions
    print("="*60)
    print("Difficulty Categories:")
    print("="*60)
    for category, (low, high) in classifier.get_difficulty_stats().items():
        print(f"  {category:12} {low:4} - {high:4}")


if __name__ == "__main__":
    main()
