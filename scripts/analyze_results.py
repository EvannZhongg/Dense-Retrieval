import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dense_retrieval.analysis import summarize_results
ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--output", default="results/summary.json"); args = ap.parse_args()
print(json.dumps(summarize_results(args.results, args.output), indent=2, default=str))

