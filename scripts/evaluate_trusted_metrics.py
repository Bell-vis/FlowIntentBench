#!/usr/bin/env python3
"""Compatibility entrypoint for the generic evidence-only evaluator."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_model_answers import *  # legacy helper imports

if __name__ == "__main__":
    raise SystemExit(main())
