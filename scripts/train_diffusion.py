"""Explicit bounded diffusion training; original checkpoints remain untouched."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finetune_diffusion import train, parser
if __name__ == '__main__':
    print(train(parser().parse_args()))
