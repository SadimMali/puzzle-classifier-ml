# Chess Puzzle Difficulty Classifier

A fast, CPU-based machine learning model that predicts chess puzzle difficulty with **±285 rating point accuracy** at **43,900 predictions/second**.

## Quick Start

```bash
pip install -r requirements.txt
python demo_server.py
# Open http://localhost:5000
```

## Deploying On Render

Render can host the FastAPI classifier as a normal Python web service.

1. Create a new Render Web Service from this repository.
2. Use the Python environment.
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to `python server.py`.
5. Set `PUZZLE_CLASSIFIER_MODEL_PATH` to the model file path available in the service filesystem.
6. Set `PUZZLE_CLASSIFIER_URL` in the backend to the Render service URL.

If the model file is not committed to git, store it in object storage, a release asset, or another persistent location and download it during build or mount it at runtime.

## Features

- **Fast**: 43,900 puzzles/second on CPU
- **Accurate**: ±285 rating points (trained on 200K Lichess puzzles)
- **5 Difficulty Levels**: Very Easy, Easy, Medium, Hard, Extreme
- **Simple API**: Single function call
- **Web Demo**: Interactive HTML interface

## Usage

### Python API

```python
from classifier import PuzzleClassifier

classifier = PuzzleClassifier()

# With solution moves (more accurate: ±285 pts)
difficulty, rating = classifier.classify(
    fen="6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - -",
    moves="e1e8"
)
print(f"{difficulty}: {rating:.0f}")  # Very Easy: 765

# Without solution moves (less accurate: ±450 pts)
difficulty, rating = classifier.classify(
    fen="6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - -"
)
```

### Batch Processing

```python
puzzles = [
    ("fen1", "e2e4 e7e5"),
    ("fen2", "g1f3 b8c6"),
]

results = classifier.classify_batch(puzzles)
```

### Web Demo

```bash
python demo_server.py
# Visit http://localhost:5001
```

Features:
- Interactive puzzle classification
- Pre-loaded examples
- Real-time predictions
- Performance stats

## How It Works

The model uses **54 features** from three sources:

1. **Question Position** (24 features): Material, mobility, king safety, pawn structure, game phase
2. **Answer Position** (24 features): Same features for the final position
3. **Solution** (6 features): Move count, captures, checks, sacrifices

**Random Forest** → **Rating Prediction** → **Difficulty Category**

## Difficulty Categories

| Category | Rating Range | Description |
|----------|--------------|-------------|
| Very Easy | 0 - 800 | Basic patterns, mate in 1-2 |
| Easy | 800 - 1200 | Simple tactics |
| Medium | 1200 - 1800 | Multi-move sequences |
| Hard | 1800 - 2400 | Complex combinations |
| Extreme | 2400+ | Advanced tactics |

## Model Performance

Trained on **200,000 Lichess puzzles**:

- **Test MAE**: ±285.6 rating points
- **Test RMSE**: ±358.2 rating points
- **Throughput**: 43,900 predictions/sec (CPU)
- **Latency**: 0.023ms per prediction

## Training Your Own Model

### 1. Download Data

```bash
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
```

### 2. Train

```bash
python train.py \
    --puzzle-db lichess_db_puzzle.csv.zst \
    --samples 200000 \
    --output models/my_model.pkl
```

Training takes ~15-20 minutes for 200K puzzles on a modern CPU.

## Project Structure

```
puzzle-classifier/
├── classifier.py          # Main API
├── train.py              # Training script
├── feature_extractor.py  # Feature engineering
├── demo_server.py        # Flask web server
├── demo.html            # Web UI
├── requirements.txt      # Dependencies
├── LICENSE              # MIT License
└── README.md           # This file
```

## Requirements

```
scikit-learn>=1.0.0
python-chess>=1.9.0
numpy>=1.21.0
tqdm>=4.62.0
flask>=2.0.0
flask-cors>=3.0.0
```

## API Reference

### `PuzzleClassifier`

```python
classifier = PuzzleClassifier(model_path="models/puzzle_rating_model_200k.pkl")
```

**Methods:**

- `classify(fen, solution_moves=None)` → `(difficulty, rating)`
- `predict_rating(fen, solution_moves)` → `rating`
- `classify_batch(puzzles)` → `[(difficulty, rating), ...]`

## Limitations

- **±285 pts error**: Good for bucketing, not exact ratings
- **Position-based**: Doesn't consider puzzle themes/tags
- **Lichess ratings**: Calibrated to Lichess puzzle database
- **CPU only**: Not optimized for GPU

## Use Cases

- Puzzle book generation
- Chess training apps
- Database filtering
- Difficulty analysis
- Adaptive learning systems

## Note on Model File

The pre-trained model (`.pkl` file) is not included in this repository due to size constraints. To use the classifier:

1. **Train your own model** using the training script (recommended)
2. **Or contact the repository owner** for access to the pre-trained model

## Acknowledgments

- **Lichess.org** for the open puzzle database and eval database - making chess data freely available to the community
- **python-chess** library for chess logic
- **scikit-learn** for ML framework
- Special thanks to the Lichess team for maintaining high-quality, open-source chess databases

## License

MIT License - See LICENSE file for details

## Citation

<!-- ```bibtex
@software{puzzle_classifier,
  title={Chess Puzzle Difficulty Classifier},
  author={sadim},
  year={2026},
  url={https://github.com/sadimmali/puzzler-classifier}
}
``` -->

---

**Made with ♟️ for the chess community**
