"""Predictor tests: documented completions, then top-1 accuracy on unseen identifiers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predict"))

from predictor import NgramPredictor  # noqa: E402

DOCUMENTED = {"Linu": "x", "Toke": "n"}
HOLDOUT = {
    "Kubernete": "s",
    "Dockerf": "i",
    "Javascrip": "t",
    "PostgreS": "Q",
    "Photosh": "o",
    "TypeScrip": "t",
    "TensorFl": "o",
    "Visual Stu": "d",
    "Chromiu": "m",
    "Kafk": "a",
    "Androi": "d",
    "Windo": "w",
}


def main() -> int:
    predictor = NgramPredictor()

    for prefix, expected in DOCUMENTED.items():
        rows = predictor.predict(prefix, 3)
        actual = rows[0][0] if rows else ""
        assert actual == expected, f"{prefix!r} -> {actual!r}, expected {expected!r}"
        print(f"documented  {prefix!r:12s} -> {actual!r} p={rows[0][1]:.3f}")

    hits = 0
    for prefix, expected in HOLDOUT.items():
        rows = predictor.predict(prefix, 1)
        actual = rows[0][0] if rows else ""
        hits += actual == expected
        print(f"holdout     {prefix!r:12s} -> {actual!r:3s} expected {expected!r}")
    accuracy = hits / len(HOLDOUT)
    print(json.dumps({"holdout_top1_accuracy": round(accuracy, 3), "n": len(HOLDOUT)}))
    assert accuracy >= 0.8, f"predictor accuracy {accuracy:.2f} below 0.8"
    print("predictor tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
