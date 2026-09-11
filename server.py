#!/usr/bin/env python3
"""
FastAPI server for puzzle classifier demo (replacement for Flask demo_server.py).

Run:
	python server.py

Opens on http://localhost:5001
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from recommender import recommend
from datetime import datetime
import uvicorn
from classifier import PuzzleClassifier
import os


class ClassifyRequest(BaseModel):
	fen: str
	moves: str | None = None


class PuzzleContent(BaseModel):
    id: str
    themes: list[str]
    puzzle_type: str | None = None
    difficulty_rating: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    class Config:
        extra = "forbid"


class HistoryContent(PuzzleContent):
    lastAttemptAt: datetime | None = None


class RecommendRequest(BaseModel):
    catalog: list[PuzzleContent]
    history: list[HistoryContent]
    candidates: list[PuzzleContent]
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)

    class Config:
        extra = "forbid"


app = FastAPI(title="Chess Puzzle Classifier Demo")

# Allow CORS for local testing (matches the previous Flask demo behavior)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


print("Loading classifier...")
classifier = PuzzleClassifier()
print("Classifier ready!")


@app.get("/")
async def index():
	"""Serve the demo HTML page."""
	demo_path = os.path.join(os.path.dirname(__file__), "demo.html")
	if not os.path.exists(demo_path):
		raise HTTPException(status_code=404, detail="demo.html not found")
	return FileResponse(demo_path, media_type="text/html")


@app.post("/classify")
async def classify(req: ClassifyRequest):
	"""Classify a puzzle.

	Expects JSON: { "fen": "...", "moves": "..." }
	"""
	fen = (req.fen or "").strip()
	moves = (req.moves or "")

	if not fen:
		raise HTTPException(status_code=400, detail="FEN is required")

	try:
		print(f"\n[Classify Request] FEN: {fen[:50]}... | Moves: {moves}")

		if moves:
			difficulty, rating = classifier.classify(fen, moves)
			has_solution = True
		else:
			difficulty, rating = classifier.classify(fen)
			has_solution = False

		result = {
			"difficulty": difficulty,
			"rating": round(rating),
			"has_solution": has_solution,
			"warning": None if has_solution else "Prediction without solution moves is less accurate (±450 pts vs ±285 pts)",
		}

		print(f"[Classify Response] Difficulty: {difficulty} | Rating: {round(rating)}")

		response = JSONResponse(result)
		response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
		response.headers["Pragma"] = "no-cache"
		response.headers["Expires"] = "0"
		return response

	except Exception as e:
		print(f"[Error] {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend_puzzles(req: RecommendRequest):
    """Internal backend contract: puzzle content in, ranked IDs and explanations out."""
    return {"items": recommend(
        [p.model_dump() for p in req.catalog],
        [p.model_dump(mode="json") for p in req.history],
        [p.model_dump() for p in req.candidates], req.offset, req.limit,
    )}


@app.get("/examples")
async def examples():
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
	return JSONResponse(examples)


@app.get("/stats")
async def stats():
	"""Get model stats."""
	return JSONResponse({
		'test_mae': 285.6,
		'test_rmse': 358.2,
		'throughput': '43,900 puzzles/sec',
		'training_samples': '200,000',
		'features': 54,
		'categories': classifier.get_difficulty_stats()
	})


if __name__ == '__main__':
	port = int(os.getenv("PORT", "5001"))
	print("\n" + "="*60)
	print("Chess Puzzle Classifier - Demo Server (FastAPI)")
	print("="*60)
	print("\n🚀 Starting server...")
	print(f"📱 Open in browser: http://localhost:{port}")
	print("⏹️  Press Ctrl+C to stop\n")

	uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")

