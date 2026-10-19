"""Shared configuration and strict checkpoint construction."""
from pathlib import Path
import hashlib
import random
import numpy as np
import torch
import yaml
from src.models.vae import MedicalAutoencoderKL
from src.models.slice_encoder import SharedSliceEncoder
from src.models.unet_diffusion import ConditionalUNet2D
from src.models.scheduler import DiffusionNoiseScheduler
from src.models.pipeline import SparseToDenseDiffusionPipeline

ROOT = Path(__file__).resolve().parents[2]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_configs(default='configs/default_config.yaml', vae='configs/vae_config.yaml', diffusion='configs/diffusion_config.yaml'):
    return tuple(yaml.safe_load((ROOT / p).read_text(encoding='utf-8')) for p in (default, vae, diffusion))

def checkpoint_id(path):
    p = Path(path)
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(p.resolve()), 'sha256': digest.hexdigest(), 'bytes': p.stat().st_size}

def load_checkpoint(path, device='cpu'):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Trained checkpoint required: {p}')
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(p, map_location=device, weights_only=True)
    if not isinstance(state, dict) or 'models' not in state:
        raise ValueError(f'Checkpoint must contain named model states: {p}')
    return state

def build_pipeline(vae_checkpoint, diffusion_checkpoint=None, device=None, weights='raw', configs=None):
    default, vc, dc = read_configs() if configs is None else configs
    device = device or (default['project']['device'] if torch.cuda.is_available() else 'cpu')
    channels = 4 if default['data']['modality'] == 'all' else 1
    if default['data']['in_channels'] != channels:
        raise ValueError('modality and in_channels disagree')
    if dc['unet'].get('conditioning_mode', 'hybrid') != 'hybrid' or dc['diffusion'].get('prediction_type', 'epsilon') != 'epsilon' or dc['diffusion'].get('sampling_method', 'ddim') != 'ddim':
        raise ValueError('Supports hybrid conditioning, epsilon prediction and DDIM only')
    vm, ec, uc = vc['model'], dc['encoder'], dc['unet']
    if ec.get('backbone', 'residual_convnet') != 'residual_convnet':
        raise ValueError('Encoder backbone must be residual_convnet')
    if ec.get('out_context_dim', uc['context_dim']) != uc['context_dim']:
        raise ValueError('Encoder and U-Net context dimensions must agree')
    if ec.get('in_channels', channels) != channels:
        raise ValueError('Encoder channels must match the selected modality')
    if default['sparsification'].get('context_window', 1) != 1 or not default['sparsification'].get('normalize_depth', True):
        raise ValueError('Supports one bounding slice per side and normalized target depth')
    if uc['in_channels'] != vm['latent_channels'] or uc['out_channels'] != vm['latent_channels']:
        raise ValueError('U-Net channels must match VAE latents')
    if vm['in_channels'] != channels or vm['out_channels'] != channels:
        raise ValueError('VAE channels must match the selected modality')
    if len(vm['channel_multipliers']) != 4:
        raise ValueError('The shared encoder requires an eightfold VAE spatial reduction')
    vae = MedicalAutoencoderKL(in_channels=channels, out_channels=channels,
        latent_channels=vm['latent_channels'], base_channels=vm['base_channels'],
        channel_multipliers=vm['channel_multipliers'], num_res_blocks=vm['num_res_blocks'],
        scale_factor=vm.get('latent_scale_factor', .18215)).to(device)
    vae.load_state_dict(load_checkpoint(vae_checkpoint)['models']['vae'], strict=True)
    vae.requires_grad_(False).eval()
    if ec.get('pos_emb_type', 'fourier') != 'fourier' or ec.get('pretrained', False) or ec.get('num_context_tokens', 16) != 16:
        raise ValueError('Encoder supports Fourier embeddings, no pretrained weights and 16 spatial tokens')
    encoder = SharedSliceEncoder(in_channels=channels, base_channels=ec.get('base_channels', 32),
        context_dim=uc['context_dim'], num_context_tokens=16,
        out_spatial_channels=ec.get('spatial_channels', 8)).to(device)
    unet = ConditionalUNet2D(in_channels=uc['in_channels'], out_channels=uc['out_channels'],
        spatial_context_channels=ec.get('spatial_channels', 8), base_channels=uc['base_channels'],
        channel_multipliers=uc['channel_multipliers'], num_res_blocks=uc['num_res_blocks'],
        context_dim=uc['context_dim'], num_heads=uc['num_heads'], num_groups=uc.get('norm_num_groups', 16)).to(device)
    if weights not in ('raw', 'ema'):
        raise ValueError('weights must be raw or ema')
    if diffusion_checkpoint is not None:
        ds = load_checkpoint(diffusion_checkpoint)
        encoder.load_state_dict(ds['models']['slice_encoder'], strict=True)
        unet.load_state_dict(ds['models']['unet'], strict=True)
        if weights == 'ema':
            if 'ema' not in ds or set(ds['ema']) != set(dict(unet.named_parameters())):
                raise ValueError('Compatible EMA weights unavailable')
            current = unet.state_dict()
            current.update(ds['ema'])
            unet.load_state_dict(current, strict=True)
            if 'ema_encoder' in ds:
                if set(ds['ema_encoder']) != set(dict(encoder.named_parameters())):
                    raise ValueError('Compatible encoder EMA weights unavailable')
                current = encoder.state_dict()
                current.update(ds['ema_encoder'])
                encoder.load_state_dict(current, strict=True)
    d = dc['diffusion']
    scheduler = DiffusionNoiseScheduler(num_train_timesteps=d['num_train_timesteps'],
        beta_schedule=d['beta_schedule'], beta_start=d.get('beta_start', .0001), beta_end=d.get('beta_end', .02)).to(device)
    scheduler.get_ddim_timesteps(d.get('ddim_sampling_steps', 25))
    if not np.isfinite(d.get('ddim_eta', 0.0)) or d.get('ddim_eta', 0.0) < 0:
        raise ValueError('DDIM eta must be finite and nonnegative')
    return SparseToDenseDiffusionPipeline(encoder, vae, unet, scheduler,
        default_inference_steps=d.get('ddim_sampling_steps', 25), default_eta=d.get('ddim_eta', 0.0)).eval()
