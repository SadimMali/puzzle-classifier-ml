"""
Train the puzzle difficulty classifier on Lichess puzzle database.

Download the puzzle database from: https://database.lichess.org/
File: lichess_db_puzzle.csv.zst
"""

import csv
import subprocess
import numpy as np
import chess
from typing import Iterator, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import time
import argparse
from tqdm import tqdm
from feature_extractor import FeatureExtractor


def parse_puzzle_csv(file_path: str, limit: int = None) -> Iterator[Dict]:
    """Parse compressed puzzle CSV database."""
    count = 0
    cmd = ["zstd", "-d", "-c", file_path]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    
    try:
        reader = csv.DictReader(process.stdout)
        for row in reader:
            yield row
            count += 1
            if limit and count >= limit:
                break
    finally:
        process.terminate()
        process.wait()


def extract_solution_features(board: chess.Board, moves_uci: str) -> np.ndarray:
    """Extract features from the puzzle solution."""
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


def get_final_position_fen(start_fen: str, moves_uci: str) -> str:
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


def prepare_dataset(puzzle_file: str, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from puzzles.
    
    Returns:
        X: features (num_samples, 54)
        y: ratings (num_samples,)
    """
    print(f"Preparing dataset with {num_samples:,} puzzles...")
    print("Features: Question + Answer + Solution (54 features)")
    
    extractor = FeatureExtractor()
    X_list = []
    y_list = []
    
    start_time = time.time()
    
    for puzzle in tqdm(parse_puzzle_csv(puzzle_file, limit=num_samples), 
                       total=num_samples, 
                       desc="Processing puzzles",
                       unit="puzzle"):
        try:
            start_fen = puzzle['FEN']
            moves = puzzle['Moves']
            rating = int(puzzle['Rating'])
            
            # Extract features
            question_features = extractor.extract_features(start_fen)
            final_fen = get_final_position_fen(start_fen, moves)
            answer_features = extractor.extract_features(final_fen)
            board = chess.Board(start_fen)
            solution_features = extract_solution_features(board, moves)
            
            features = np.concatenate([
                question_features,
                answer_features,
                solution_features
            ])
            
            X_list.append(features)
            y_list.append(rating)
            
        except:
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    prep_time = time.time() - start_time
    
    print(f"\nDataset prepared in {prep_time/60:.1f} minutes")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Rating: {y.min():.0f} - {y.max():.0f} (mean: {y.mean():.0f})")
    
    return X, y


def train_model(X_train, y_train, X_test, y_test, n_estimators=200, max_depth=25):
    """Train Random Forest model."""
    print("\nTraining Random Forest...")
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nTest Results:")
    print(f"  MAE:  {test_mae:.1f} rating points")
    print(f"  RMSE: {test_rmse:.1f} rating points")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train puzzle difficulty classifier')
    parser.add_argument('--puzzle-db', type=str, required=True,
                       help='Path to lichess_db_puzzle.csv.zst')
    parser.add_argument('--samples', type=int, default=200000,
                       help='Number of puzzles to train on (default: 200000)')
    parser.add_argument('--output', type=str, default='models/puzzle_rating_model.pkl',
                       help='Output model path')
    parser.add_argument('--test-size', type=float, default=0.1,
                       help='Test set fraction (default: 0.1)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Chess Puzzle Difficulty Classifier - Training")
    print("="*60)
    
    # Prepare dataset
    X, y = prepare_dataset(args.puzzle_db, args.samples)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]:,} samples")
    print(f"Test set:  {X_test.shape[0]:,} samples")
    
    # Train
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {args.output}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
