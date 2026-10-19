"""Bounded fine-tuning with equal expected probability for each training stride."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from src.models.factory import ROOT, read_configs, build_pipeline, seed_everything, load_checkpoint, checkpoint_id
from src.data.dataset import BraTSSliceDataset, configured_patient_splits
from src.utils.checkpoint import EMAModel

def train(args):
    if args.max_steps < 1 or args.time_limit_minutes <= 0 or args.batch_size < 1:
        raise ValueError('Steps, time limit and batch size must be positive')
    seed_everything(args.seed)
    started = time.monotonic()
    deadline = started + args.time_limit_minutes * 60
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / 'latest.pt').exists():
        raise FileExistsError('Output contains a run; use --resume or a new directory')
    configs = read_configs(diffusion=args.config)
    default, _, dc = configs
    data_dir = str(ROOT / default['data']['data_dir'])
    train_vols, val_vols, test_vols = configured_patient_splits(default, seed=args.seed)
    if args.smoke:
        train_vols = train_vols[:1]
    init = ROOT / (args.resume or args.init_checkpoint)
    vae_path = ROOT / dc['training']['vae_checkpoint']
    pipeline = build_pipeline(vae_path, init, configs=configs)
    device = next(pipeline.parameters()).device
    params = list(pipeline.slice_encoder.parameters()) + list(pipeline.unet.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=dc['training']['weight_decay'])
    amp_enabled = device.type == 'cuda' and dc['training'].get('mixed_precision', True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    ema = EMAModel(pipeline.unet, dc['training']['ema_decay'])
    ema_enc = EMAModel(pipeline.slice_encoder, dc['training']['ema_decay'])
    step = 0
    sampler_gen = torch.Generator().manual_seed(args.seed)
    if args.resume:
        state = load_checkpoint(init, device)
        optimizer.load_state_dict(state['optimizers']['optimizer'])
        scaler.load_state_dict(state['scalers']['scaler'])
        ema.load_state_dict(state['ema'])
        ema_enc.load_state_dict(state['ema_encoder'])
        step = state['step']
        sampler_gen.set_state(state['sampler_rng'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    datasets = [BraTSSliceDataset(data_dir, train_vols, default['data']['modality'],
        stride=s, total_slices_per_vol=default['data']['total_slices'], augment=False) for s in args.strides]
    if any(len(ds) == 0 for ds in datasets):
        raise ValueError('Training dataset is empty')
    combined = ConcatDataset(datasets)
    weights = torch.cat([torch.full((len(ds),), 1.0 / len(ds), dtype=torch.double) for ds in datasets])
    # Draw a batch at a time so checkpointed generator state denotes the next batch.
    metadata = {'arguments': vars(args), 'train_volumes': train_vols, 'validation_volumes': val_vols,
        'test_volumes': test_vols, 'configs': configs, 'initial_checkpoint': checkpoint_id(init),
        'vae_checkpoint': checkpoint_id(vae_path), 'normalization': 'acquired-only per volume and stride',
        'stride_sampling': 'equal probability; replacement', 'torch_version': str(torch.__version__),
        'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None}
    (out / ('resume_manifest.json' if args.resume else 'manifest.json')).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logpath = out / 'training.csv'
    log = logpath.open('a', newline='', encoding='utf-8')
    writer = csv.DictWriter(log, fieldnames=['step','loss','elapsed_seconds','lr'])
    if logpath.stat().st_size == 0:
        writer.writeheader()
    from torch.utils.data._utils.collate import default_collate
    pipeline.slice_encoder.train()
    pipeline.unet.train()
    last_loss = float('nan')
    def save():
        state = {'epoch': 0, 'step': step, 'metric_value': last_loss,
            'models': {'slice_encoder': pipeline.slice_encoder.state_dict(), 'unet': pipeline.unet.state_dict()},
            'optimizers': {'optimizer': optimizer.state_dict()}, 'scalers': {'scaler': scaler.state_dict()},
            'ema': ema.state_dict(), 'ema_encoder': ema_enc.state_dict(),
            'sampler_rng': sampler_gen.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'review_metadata': metadata}
        temp = out / 'latest.tmp.pt'
        torch.save(state, temp)
        temp.replace(out / 'latest.pt')
    try:
        while step < args.max_steps and time.monotonic() < deadline:
            indices = torch.multinomial(weights, args.batch_size, replacement=True, generator=sampler_gen).tolist()
            batch = default_collate([combined[i] for i in indices])
            if time.monotonic() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp_enabled):
                with torch.no_grad():
                    clean = pipeline.vae.encode(batch['target'].to(device)).sample() * pipeline.vae.scale_factor
                noise = torch.randn_like(clean)
                t = torch.randint(0, pipeline.scheduler.num_train_timesteps, (len(indices),), device=device)
                noisy = pipeline.scheduler.add_noise(clean, noise, t)
                tokens, spatial = pipeline.slice_encoder(batch['prev'].to(device), batch['next'].to(device),
                    batch['normalized_z'].to(device), batch['alpha'].to(device), batch['gap_dist'].to(device))
                prediction = pipeline.unet(noisy, t, tokens, spatial)
                loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, dc['training']['grad_clip_norm'])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                ema.update(pipeline.unet)
                ema_enc.update(pipeline.slice_encoder)
            step += 1
            last_loss = float(loss.detach())
            writer.writerow({'step': step, 'loss': last_loss, 'elapsed_seconds': time.monotonic()-started, 'lr': optimizer.param_groups[0]['lr']})
            if step % 25 == 0 or args.smoke:
                log.flush()
                print(f'step={step} loss={last_loss:.5f} seconds={time.monotonic()-started:.1f}', flush=True)
            if step % 500 == 0:
                save()
    finally:
        save()
        log.close()
        (out / 'completion.json').write_text(json.dumps({'step': step, 'seconds': time.monotonic()-started,
            'stopped_by': 'step_limit' if step >= args.max_steps else 'time_limit_or_interruption'}, indent=2))
    return str(out / 'latest.pt')

def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--init-checkpoint', default='experiments/checkpoints/diffusion/best_model.pt')
    p.add_argument('--resume', help='Explicit resume checkpoint; optimizer and RNG state restored')
    p.add_argument('--config', default='configs/diffusion_config.yaml')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--strides', type=int, nargs='+', default=[4,8,16])
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--time-limit-minutes', type=float, default=70)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output-dir', default='experiments/review/finetune')
    p.add_argument('--smoke', action='store_true')
    return p

if __name__ == '__main__':
    print(train(parser().parse_args()))
