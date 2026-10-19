"""Compatibility entry point for the corrected benchmark."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_review import run_benchmark, main
if __name__ == '__main__':
    main()
