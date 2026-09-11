"""Extract features from chess positions for ML model."""

import chess
import numpy as np


class FeatureExtractor:
    """Extract features from chess positions."""
    
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def extract_features(self, fen: str) -> np.ndarray:
        """
        Extract features from a position.
        
        Features:
        - Material balance (1)
        - Piece counts for each side (12)
        - King safety metrics (4)
        - Piece mobility approximation (2)
        - Center control (1)
        - Pawn structure (3)
        - Phase of game (1)
        - Checks/attacks (2)
        Total: ~26 features
        """
        board = chess.Board(fen)
        features = []
        
        # Material balance
        material = self._material_balance(board)
        features.append(material)
        
        # Piece counts for each side
        white_pieces = self._piece_counts(board, chess.WHITE)
        black_pieces = self._piece_counts(board, chess.BLACK)
        features.extend(white_pieces)
        features.extend(black_pieces)
        
        # King safety
        white_king_safety = self._king_safety(board, chess.WHITE)
        black_king_safety = self._king_safety(board, chess.BLACK)
        features.extend([white_king_safety, black_king_safety])
        
        # Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))
        features.append(mobility)
        
        # Opponent mobility
        board.push(chess.Move.null())
        opp_mobility = len(list(board.legal_moves)) if not board.is_check() else 0
        board.pop()
        features.append(opp_mobility)
        
        # Center control (pieces in center)
        center_control = self._center_control(board)
        features.append(center_control)
        
        # Pawn structure
        pawn_features = self._pawn_structure(board)
        features.extend(pawn_features)
        
        # Game phase (0=opening, 1=endgame)
        phase = self._game_phase(board)
        features.append(phase)
        
        # Checks and attacks
        in_check = 1 if board.is_check() else 0
        features.append(in_check)
        
        # Number of attacked pieces
        attacked = self._count_attacked_pieces(board)
        features.append(attacked)
        
        return np.array(features, dtype=np.float32)
    
    def _material_balance(self, board: chess.Board) -> float:
        """Calculate material balance from white's perspective."""
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                balance += value if piece.color == chess.WHITE else -value
        return balance
    
    def _piece_counts(self, board: chess.Board, color: chess.Color) -> list:
        """Count pieces for a given side."""
        counts = []
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                          chess.ROOK, chess.QUEEN, chess.KING]:
            count = len(board.pieces(piece_type, color))
            counts.append(count)
        return counts
    
    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """Estimate king safety (number of friendly pieces around king)."""
        king_square = board.king(color)
        if king_square is None:
            return 0
        
        # Count friendly pieces around king
        safety = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                rank = chess.square_rank(king_square) + dy
                file = chess.square_file(king_square) + dx
                if 0 <= rank < 8 and 0 <= file < 8:
                    sq = chess.square(file, rank)
                    piece = board.piece_at(sq)
                    if piece and piece.color == color:
                        safety += 1
        return safety
    
    def _center_control(self, board: chess.Board) -> float:
        """Count pieces in center squares."""
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        count = 0
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece:
                count += 1 if piece.color == board.turn else -1
        return count
    
    def _pawn_structure(self, board: chess.Board) -> list:
        """Analyze pawn structure."""
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Doubled pawns
        white_doubled = self._count_doubled_pawns(white_pawns)
        black_doubled = self._count_doubled_pawns(black_pawns)
        
        # Passed pawns
        passed = len(white_pawns) - len(black_pawns)
        
        return [white_doubled, black_doubled, passed]
    
    def _count_doubled_pawns(self, pawns) -> int:
        """Count doubled pawns."""
        files = [chess.square_file(sq) for sq in pawns]
        return len(files) - len(set(files))
    
    def _game_phase(self, board: chess.Board) -> float:
        """Estimate game phase (0=opening, 1=endgame)."""
        # Count major pieces
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + 
                 len(board.pieces(chess.KNIGHT, chess.BLACK)) +
                 len(board.pieces(chess.BISHOP, chess.WHITE)) + 
                 len(board.pieces(chess.BISHOP, chess.BLACK)))
        
        total_material = queens * 9 + rooks * 5 + minors * 3
        max_material = 2 * 9 + 4 * 5 + 8 * 3  # Starting material
        
        return 1.0 - (total_material / max_material)
    
    def _count_attacked_pieces(self, board: chess.Board) -> int:
        """Count number of attacked pieces."""
        attacked = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    attacked += 1
        return attacked


if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    ]
    
    for fen in test_fens:
        features = extractor.extract_features(fen)
        print(f"FEN: {fen[:40]}...")
        print(f"Features shape: {features.shape}, First 5: {features[:5]}")
        print()
