#!/usr/bin/env python3
"""
Simple Flask server for puzzle classifier demo.

Usage:
    python demo_server.py
    
Then open: http://localhost:5001
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from classifier import PuzzleClassifier
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for local testing

# Load classifier once at startup
print("Loading classifier...")
classifier = PuzzleClassifier()
print("Classifier ready!")

@app.route('/')
def index():
    """Serve the demo HTML page."""
    return send_file('demo.html')

@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify a puzzle.
    
    Expected JSON:
    {
        "fen": "position FEN",
        "moves": "e2e4 e7e5" (optional)
    }
    
    Returns JSON:
    {
        "difficulty": "Medium",
        "rating": 1638,
        "has_solution": true
    }
    """
    try:
        data = request.json
        fen = data.get('fen', '').strip()
        moves = data.get('moves', '').strip()
        
        print(f"\n[Classify Request] FEN: {fen[:50]}... | Moves: {moves}")
        
        if not fen:
            return jsonify({'error': 'FEN is required'}), 400
        
        # Classify
        if moves:
            difficulty, rating = classifier.classify(fen, moves)
            has_solution = True
        else:
            difficulty, rating = classifier.classify(fen)
            has_solution = False
        
        result = {
            'difficulty': difficulty,
            'rating': round(rating),
            'has_solution': has_solution,
            'warning': None if has_solution else 'Prediction without solution moves is less accurate (±450 pts vs ±285 pts)'
        }
        
        print(f"[Classify Response] Difficulty: {difficulty} | Rating: {round(rating)}")
        
        response = jsonify(result)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        print(f"[Error] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/examples', methods=['GET'])
def examples():
    """Get example puzzles."""
    examples = [
        {
            'name': 'Very Easy: Mate in 1',
            'fen': '6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - -',
            'moves': 'e1e8',
            'expected': 'Very Easy (~700)'
        },
        {
            'name': 'Easy: Simple Fork',
            'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -',
            'moves': 'f3e5 c6e5 c4f7',
            'expected': 'Medium (~1600)'
        },
        {
            'name': 'Medium: Knight Fork',
            'fen': 'r1bqk2r/ppp2ppp/2n5/3p4/1b1Pn3/2N2N2/PPP2PPP/R1BQKB1R w KQkq -',
            'moves': 'f3e5 c6e5 d1d5',
            'expected': 'Medium (~1400)'
        },
        {
            'name': 'Hard: Complex Sacrifice',
            'fen': 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - -',
            'moves': 'c4f7 f8f7 c3d5 d8d7 d5f6 g7f6',
            'expected': 'Medium (~1600)'
        },
        {
            'name': 'Position Only (No Solution)',
            'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -',
            'moves': '',
            'expected': '~1000 (less accurate)'
        }
    ]
    return jsonify(examples)

@app.route('/stats', methods=['GET'])
def stats():
    """Get model stats."""
    return jsonify({
        'test_mae': 285.6,
        'test_rmse': 358.2,
        'throughput': '43,900 puzzles/sec',
        'training_samples': '200,000',
        'features': 54,
        'categories': classifier.get_difficulty_stats()
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Chess Puzzle Classifier - Demo Server")
    print("="*60)
    print("\n🚀 Starting server...")
    print("📱 Open in browser: http://localhost:5001")
    print("⏹️  Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
