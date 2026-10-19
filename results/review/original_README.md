# Sparse-to-Dense 3D Medical Volume Reconstruction via Conditional Latent Diffusion

A complete end-to-end deep learning pipeline that reconstructs missing/unsampled MRI slices from a sparse axial stack using a **Conditional Latent Diffusion Model (CLDM)**, then assembles the synthesized slices back into a dense, anatomically consistent 3D brain volume.

---

## Architecture

```
Sparse 2D Slices (I_prev, I_next)
        │
        ▼
[Weight-Tied Contextual Slice Encoder (ResNet)]
        + [Continuous Depth Positional Embedding P(z, alpha, gap)]
        │
        ├─── Context Tokens  (B, 17, 256) ──────────────────────┐
        └─── Spatial Context (B, 8,  30, 30) ───────────────────┤
                                                                 │
[Target Slice] ──► [VAE Encoder] ──► [Clean Latent z_0]         │
                                          │                      │
                         + Gaussian Noise (T steps)             │
                                          │                      │
                              [Noisy Latent z_t]                 │
                                          │                      │
                              [Conditional Denoising U-Net] ◄───┘
                              (Cross-Attention on Context Tokens)
                                          │
                         ── DDIM Reverse Sampling (25 steps) ──►
                                          │
                              [Denoised Latent z_0*]
                                          │
                              [VAE Decoder] ──► [Reconstructed Slice]
                                          │
                [3D Volume Assembly + Inter-slice Smoothing]
                                          │
                    [Dense 3D Volume (155 × 240 × 240)]
                                          │
         [Evaluation: PSNR / SSIM / MAE / MSE / Dice | PyVista 3D Rendering]
```

---

## Environment Setup

### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA 12.8+ (tested on RTX 5060 Laptop GPU)
- PyTorch 2.9.1+cu128

### Installation

```bash
# Clone the project directory
cd "Project 1"

# Install all dependencies using the dedicated environment
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" -m pip install -r requirements.txt
```

All primary dependencies:
| Package | Purpose |
|---------|---------|
| `torch>=2.0` | Core deep learning framework |
| `monai` | Medical imaging transforms & utilities |
| `nibabel` | NIfTI volume reading/writing |
| `h5py` | BraTS HDF5 slice file reading |
| `einops` | Tensor manipulation |
| `pyvista` | 3D volume rendering |
| `open3d` | 3D point cloud & mesh visualization |
| `scipy` | Interpolation baselines |
| `opencv-python` | 2D image processing |
| `tensorboard` | Training metrics dashboard |

---

## Dataset

### BraTS 2020 (Primary)
The dataset is already available locally at `./BraTS2020_training_data/content/data/`.

- **Format**: Pre-processed HDF5 slices: `volume_{ID}_slice_{z}.h5`
- **Structure**: Each `.h5` file contains:
  - `image`: `(240, 240, 4)` float64 — 4 MRI modalities [FLAIR, T1, T1ce, T2]
  - `mask`: `(240, 240, 3)` uint8 — 3 tumor class masks [WT, TC, ET]
- **Scale**: 369 volumes × 155 slices = 57,195 slice files

### Data Split (Patient-Level, No Leakage)
| Split | Volumes | Slices |
|-------|---------|--------|
| Train | 258 (70%) | ~39,990 |
| Val | 55 (15%) | ~8,525 |
| Test | 56 (15%) | ~8,680 |

---

## Configuration

All hyperparameters are managed via YAML config files in `configs/`:

| File | Controls |
|------|---------|
| `configs/default_config.yaml` | Paths, modality, splits, sparsification strides |
| `configs/vae_config.yaml` | VAE architecture & training hyperparameters |
| `configs/diffusion_config.yaml` | Encoder, U-Net, DDIM schedule, training hyperparameters |
| `configs/eval_config.yaml` | Evaluation strides, baselines, visualization settings |

### Key Settings in `default_config.yaml`
```yaml
data:
  modality: "flair"           # Primary: FLAIR | Options: t1, t1ce, t2, all
  in_channels: 1              # 1 for single-channel, 4 for 'all'

sparsification:
  default_stride: 4           # Training/default stride
  eval_strides: [2, 4, 8, 16] # Strides evaluated in benchmark
```

---

## Sparsification Strides

| Stride | Acquired Slices | Missing Slices | Use Case |
|--------|----------------|----------------|---------|
| 2× | 78 | 77 | Sanity check — easy baseline |
| 4× | 39 | 116 | **Primary benchmark** |
| 8× | 20 | 135 | Aggressive sparsity |
| 16× | 10 | 145 | Stress test |

---

## Running the Pipeline

### Step 1 — Verify Data & Run Tests
```bash
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" -m pytest tests/ -v
```
Expected: **17 passed** across data, encoder, VAE, diffusion, and metrics modules.

### Step 2 — Baseline Benchmark (No Training Required)
Evaluate classical interpolation baselines (Nearest, Linear, Cubic, B-Spline) across strides 2×, 4×, 8×, 16× on the test set:
```bash
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" scripts/evaluate_all.py --vols 5 --no_diff
```
Results saved to `./results/benchmark_metrics.csv` and `./results/benchmark_summary.md`.

### Step 3 — Train the VAE
```bash
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" scripts/train_vae.py
```
- Trains for 30 epochs (configurable in `vae_config.yaml`)
- Best checkpoint saved to `./experiments/checkpoints/vae/best_model.pt`
- Monitor training with TensorBoard: `tensorboard --logdir ./experiments/tb_logs`

### Step 4 — Train the Conditional Latent Diffusion Model
```bash
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" scripts/train_diffusion.py
```
- Trains Shared Slice Encoder + Conditional U-Net for 50 epochs (configurable)
- Requires VAE checkpoint (auto-loads from `./experiments/checkpoints/vae/best_model.pt`)
- Best checkpoint saved to `./experiments/checkpoints/diffusion/best_model.pt`
- Features: AMP (FP16), EMA weights, cosine LR scheduling, gradient clipping

### Step 5 — Full Benchmark (Baselines + Diffusion)
After training completes:
```bash
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" scripts/evaluate_all.py --vols 10
```

### Step 6 — Reconstruct a Single Volume
```bash
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" scripts/reconstruct_volume.py --vol_id 100 --stride 4 --format nifti
```
Saves as `./results/reconstructed_volumes/recon_vol_100_stride_4x.nii.gz`.

### Step 7 — Generate Visualizations
```bash
& "D:\Kaggle competetion\torch_env\Scripts\python.exe" scripts/demo_visualize.py --vol_id 100 --stride 4
```
Generates:
- 2D multi-panel comparison grids at axial slices z=50, 75, 100
- Orthogonal cross-sections (Axial, Coronal, Sagittal) with error heatmaps
- 3D PyVista volume renders (sparse vs reconstructed vs ground truth)

---

## Project Structure

```
Project 1/
├── configs/
│   ├── default_config.yaml     # Master config: paths, modality, sparsification
│   ├── vae_config.yaml         # VAE architecture & training hyperparams
│   ├── diffusion_config.yaml   # Encoder, U-Net, DDIM, training hyperparams
│   └── eval_config.yaml        # Evaluation strides, baselines, viz settings
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # BraTSSliceDataset, BraTSVolumeDataset, patient splits
│   │   ├── sparsifier.py       # Volume sparsification & gap context extraction
│   │   └── transforms.py       # Intensity normalization, spatial augmentations
│   ├── models/
│   │   ├── positional_embedding.py # Continuous Fourier depth embeddings
│   │   ├── slice_encoder.py    # Weight-tied Contextual Slice Encoder
│   │   ├── vae.py              # Medical AutoencoderKL (240→30 latent compression)
│   │   ├── unet_diffusion.py   # Conditional Denoising U-Net with Cross-Attention
│   │   ├── scheduler.py        # Cosine/Linear noise schedules, DDIM sampler
│   │   └── pipeline.py         # End-to-end sparse→dense reconstruction pipeline
│   ├── baselines/
│   │   └── interpolators.py    # Nearest, Linear, Cubic, B-Spline baselines
│   ├── evaluation/
│   │   ├── metrics.py          # PSNR, SSIM, MAE, MSE, Dice implementations
│   │   └── evaluator.py        # Volume benchmark engine
│   ├── training/
│   │   └── losses.py           # VAELoss, DiffusionNoiseLoss
│   ├── visualization/
│   │   ├── slice_viewer.py     # 2D multi-panel slice comparisons & orthogonal views
│   │   └── volume_viewer_3d.py # PyVista 3D volume rendering
│   └── utils/
│       ├── logger.py           # Structured logging & TensorBoard tracking
│       └── checkpoint.py       # CheckpointManager & EMA weights
│
├── scripts/
│   ├── train_vae.py            # VAE pre-training script
│   ├── train_diffusion.py      # Conditional diffusion training script
│   ├── evaluate_all.py         # Benchmark: baselines vs diffusion across strides
│   ├── reconstruct_volume.py   # CLI: reconstruct a single volume to NIfTI/H5
│   └── demo_visualize.py       # Generate 2D/3D comparison visualizations
│
├── tests/
│   ├── test_data.py            # Data pipeline, sparsification, patient split tests
│   ├── test_encoder.py         # Positional embedding & slice encoder tests
│   ├── test_vae.py             # VAE forward/backward, encode/decode tests
│   ├── test_diffusion.py       # Scheduler, U-Net, end-to-end pipeline tests
│   └── test_metrics.py         # PSNR, SSIM, Dice, baseline interpolator tests
│
├── results/                    # Auto-generated benchmark and visualization outputs
├── experiments/                # Checkpoints and TensorBoard logs
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## Model Architecture Details

### 1. Shared Contextual Slice Encoder
- **Backbone**: ResNet-style ConvNet (4 blocks, 240→30 spatial compression)
- **Weight-tying**: Same weights applied to both `I_prev` and `I_next` slices
- **Depth Embedding**: 3-component Fourier embedding — global depth `z/D`, interpolation `α = (z - z_prev)/(z_next - z_prev)`, gap size `Δz/16`
- **Output**: `(B, 17, 256)` cross-attention context tokens + `(B, 8, 30, 30)` spatial conditioning maps

### 2. Medical AutoencoderKL (VAE)
- **Compression ratio**: 8× spatial (`240×240 → 30×30`) with 4 latent channels
- **Architecture**: 4-level ResNet encoder/decoder with self-attention at deep levels
- **Loss**: L1 + L2 + Spatial Gradient Sharpness + KL Divergence (β = 1e-5)

### 3. Conditional Denoising U-Net
- **Conditioning**: Hybrid — **Cross-Attention** on context tokens (primary) + **spatial concatenation** of conditioning maps (secondary)
- **Timestep injection**: Sinusoidal embedding + AdaGN via ResNet time projections
- **U-Net depth**: 3 levels (30×30 → 15×15 → 8×8 latent space)

### 4. Diffusion Sampler
- **Forward process**: Cosine β-schedule, T=1000 steps
- **Inference**: DDIM deterministic sampling (η=0), 25 steps (~14× speedup over full DDPM)
- **Prediction target**: ε-prediction (noise prediction)

---

## Ablation Experiment Plan

| Experiment | Description |
|-----------|-------------|
| Stride 2×/4×/8×/16× | PSNR/SSIM/MAE vs sparsification level |
| FLAIR vs T1ce | Single-channel modality comparison |
| DDIM steps 5/10/25/50 | Inference speed vs quality tradeoff |
| With/without inter-slice smoothing | Smoothing postprocessing ablation |
| DDPM vs DDIM | Sampling method comparison |
| 4-channel joint (`modality: all`) | Multi-modal conditioning ablation |

---

## Expected Benchmark Results

After training (based on literature references):

| Method | PSNR (missing slices) | SSIM | Notes |
|--------|-----------------------|------|-------|
| Nearest Neighbor | ~22 dB | ~0.65 | Baseline lower bound |
| Linear Interp | ~27 dB | ~0.80 | Standard baseline |
| Cubic Spline | ~28 dB | ~0.82 | Strong classical baseline |
| **CLDM (Ours)** | **~31-33 dB** | **~0.89+** | Learned generative approach |

*Results will populate in `./results/benchmark_summary.md` after training + evaluation.*
