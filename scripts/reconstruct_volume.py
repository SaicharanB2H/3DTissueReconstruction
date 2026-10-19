"""Compatibility entry point for the corrected reconstruction CLI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reconstruct_review import reconstruct_single_volume, main
if __name__ == '__main__':
    main()
