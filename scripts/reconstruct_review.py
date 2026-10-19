"""Reconstruct one volume with strict weights and acquired-only preprocessing."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.models.factory import ROOT, read_configs, build_pipeline, seed_everything, checkpoint_id
from src.data.dataset import BraTSVolumeDataset
from src.data.preprocessing import normalize_acquired_volume
from src.data.sparsifier import VolumeSparsifier
from src.evaluation.metrics import evaluate_volume_reconstruction
from src.utils.volume_io import export_volume

def reconstruct_single_volume(volume_id, stride=4,
    diffusion_ckpt='experiments/checkpoints/diffusion/best_model.pt',
    vae_ckpt='experiments/checkpoints/vae/best_model.pt',
    output_dir='results/review/reconstructed_volumes', save_format='nifti', seed=42,
    weights='raw', steps=None, smoothing=True):
    seed_everything(seed)
    cfg = read_configs()[0]
    pipeline = build_pipeline(ROOT / vae_ckpt, ROOT / diffusion_ckpt, weights=weights)
    device = next(pipeline.parameters()).device
    steps = pipeline.default_inference_steps if steps is None else steps
    ds = BraTSVolumeDataset(str(ROOT / cfg['data']['data_dir']), [volume_id], cfg['data']['modality'], cfg['data']['total_slices'])
    raw = ds[0]['volume_raw']
    sp = VolumeSparsifier(stride, len(raw))
    acquired = sp.get_acquired_indices()
    gt, stats = normalize_acquired_volume(raw, acquired)
    sparse, _, _, missing = sp.sparsify_volume(gt.to(device))
    with torch.autocast(device.type, enabled=device.type == 'cuda'):
        recon = pipeline.reconstruct_volume(sparse, acquired, stride, num_inference_steps=steps,
            apply_smoothing=smoothing, generator=torch.Generator(device=device).manual_seed(seed))
    metrics = evaluate_volume_reconstruction(recon, gt.to(device), missing_indices=missing)
    print(metrics)
    suffix = '.nii.gz' if save_format == 'nifti' else '.h5'
    return export_volume(recon, ROOT / output_dir / f'recon_vol_{volume_id}_stride_{stride}x{suffix}',
        {'volume_id': volume_id, 'stride': stride, 'seed': seed, 'acquired_indices': acquired,
         'normalization_stats': stats, 'metrics': metrics, 'weights': weights, 'steps': steps,
         'eta':pipeline.default_eta,
         'smoothing': smoothing, 'diffusion_checkpoint': checkpoint_id(ROOT / diffusion_ckpt),
         'vae_checkpoint': checkpoint_id(ROOT / vae_ckpt)})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vol_id', type=int, default=100)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--format', choices=['nifti', 'h5'], default='nifti')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--weights', choices=['raw', 'ema'], default='raw')
    p.add_argument('--steps', type=int, help='Override configured DDIM steps')
    p.add_argument('--no-smoothing', action='store_true')
    p.add_argument('--diffusion-ckpt', default='experiments/checkpoints/diffusion/best_model.pt')
    p.add_argument('--vae-ckpt', default='experiments/checkpoints/vae/best_model.pt')
    p.add_argument('--output-dir', default='results/review/reconstructed_volumes')
    a = p.parse_args()
    print(reconstruct_single_volume(a.vol_id, a.stride, a.diffusion_ckpt, a.vae_ckpt,
        a.output_dir, a.format, a.seed, a.weights, a.steps, not a.no_smoothing))

if __name__ == '__main__':
    main()
