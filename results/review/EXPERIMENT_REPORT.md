# Corrected reconstruction experiment

Completed locally on 8 October 2026. Corrected original reference retained using raw weights, 25 DDIM steps, smoothing True.

Across 20 patient/stride pairs, selected versus the corrected raw 25-step smoothed reference: PSNR change +0.000 ± 0.000 dB; SSIM change +0.0000 ± 0.0000.

| Stride | Method | Missing PSNR (dB), mean ± SD | Missing SSIM, mean ± SD | Mean seconds |
|---|---|---:|---:|---:|
| 8 | Corrected reference | 28.037 ± 1.577 | 0.9054 ± 0.0121 | 7.65 |
| 8 | Selected | 28.037 ± 1.577 | 0.9054 ± 0.0121 | 7.93 |
| 8 | Linear | 27.628 ± 1.119 | 0.9226 ± 0.0097 | 0.15 |
| 8 | Smoothed linear | 27.628 ± 1.119 | 0.9225 ± 0.0097 | 0.21 |
| 16 | Corrected reference | 25.071 ± 1.076 | 0.8818 ± 0.0119 | 7.29 |
| 16 | Selected | 25.071 ± 1.076 | 0.8818 ± 0.0119 | 8.40 |
| 16 | Linear | 24.550 ± 1.242 | 0.8752 ± 0.0123 | 0.15 |
| 16 | Smoothed linear | 24.550 ± 1.242 | 0.8752 ± 0.0123 | 0.20 |

## Protocol and data separation

Seed 42. Validation patients: [2, 5, 9, 15, 28]. Final test patients: [14, 21, 22, 35, 49, 50, 51, 53, 55, 59]. The configured split is 258 training / 55 validation / 56 test volumes, disjoint by volume. Final comparisons use the ten historical benchmark patients, not the full internal test partition.

Normalization bounds come only from acquired slices for each volume and stride; context and withheld targets share these bounds. Diffusion training excludes acquired targets. The architecture and pretrained VAE remain unchanged. AdamW fine-tunes encoder/U-Net at 2e-5 with batch 8, equal-probability strides 4/8/16, weight decay 1e-4, clipping norm 1 and EMA decay .999. Augmentation is disabled. Raw/EMA, 25/50 steps and smoothing off/on yield eight configurations per model, evaluated on the same validation cases with paired patient/stride seeds.

The best corrected reference validation setting establishes the eligibility threshold. Candidates need a strictly higher mean missing-slice PSNR across both strides and non-decreasing mean SSIM. The selected candidate is tested once on the held-out cases. Final paired gains use the common raw 25-step smoothed reference; when selection chooses another reference setting, these gains can reflect setting changes rather than fine-tuning. No test cases select hyperparameters.

## Validation grid

Means across five patients and strides 8/16 (ten patient/stride pairs per setting):

| Model | Weights | DDIM steps | Smoothing | Mean PSNR | Mean SSIM |
|---|---|---:|---|---:|---:|
| Reference | ema | 25 | False | 25.7921 | 0.88694 |
| Reference | ema | 25 | True | 26.1761 | 0.89113 |
| Reference | ema | 50 | False | 25.6116 | 0.88562 |
| Reference | ema | 50 | True | 26.0494 | 0.89019 |
| Reference | raw | 25 | False | 25.8035 | 0.88697 |
| Reference | raw | 25 | True | 26.1862 | 0.89114 |
| Reference | raw | 50 | False | 25.6256 | 0.88564 |
| Reference | raw | 50 | True | 26.0624 | 0.89019 |
| Fine-tuned | ema | 25 | False | 25.6557 | 0.88498 |
| Fine-tuned | ema | 25 | True | 26.1643 | 0.88992 |
| Fine-tuned | ema | 50 | False | 25.3791 | 0.88282 |
| Fine-tuned | ema | 50 | True | 25.9618 | 0.88829 |
| Fine-tuned | raw | 25 | False | 25.6407 | 0.88482 |
| Fine-tuned | raw | 25 | True | 26.1392 | 0.88982 |
| Fine-tuned | raw | 50 | False | 25.4001 | 0.88285 |
| Fine-tuned | raw | 50 | True | 25.9668 | 0.88835 |

## Budget and runtime

Total bounded protocol: 40.36 minutes (maximum 120). Fine-tuning: 3000 steps, 18.49 minutes, stopped by step_limit.

- Corrected reference evaluation and validation grid: 10.67 minutes (20-minute allocation).
- Fine-tuning: 18.49 minutes (70-minute allocation).
- Candidate validation and final comparisons: 11.21 minutes (30-minute allocation).

Device: NVIDIA RTX 5060 Laptop GPU, approximately 8 GB VRAM; PyTorch 2.9.1+cu128, Python 3.10.10. CUDA autocast is enabled; schedule and fidelity metrics use float32. Timings include loading/metrics in stage manifests. Diffusion row times measure generation and CUDA synchronization, excluding smoothing and metric calculation. Baseline row times cover interpolation plus metrics, so the displayed seconds are not an equal-work speed comparison. The two smoothing options reuse the same generated volume; their times must not be added as independent generations. Smoke verification was separate and short.

## Evidence and reproduction

- `experiment_manifest.json`: case IDs, seed and budget.
- `test_reference.csv`, `validation_reference.csv`, `validation_candidate.csv`, `test_selected.csv`: per-patient metrics, parameters, generation times and checkpoint hashes.
- Matching JSON files: configuration snapshots, identities, precision, GPU and stage elapsed time.
- `selection.json`: actual selected settings and validation thresholds.
- `paired_comparison.csv`: each test patient's PSNR/SSIM change.
- `experiments/review/finetune/manifest.json`, `training.csv`, `completion.json`, `latest.pt`: initialization, training losses, stopping reason and resumable model/optimizer/RNG/EMA state.
- `source_checksums.json`: final source identities. Minor interface validation/documentation corrections during training do not change the protocol's default numerical settings; manifests are the configuration record for the completed run.

From the project directory, run `scripts/review_experiment.py` with the approved interpreter. Preserve these outputs first and use separate paths for another run. No original checkpoint or historical CSV was replaced.

## Interpretation and limitations

These are retrospective image-fidelity measurements including background. Standard deviations describe variability across ten patients, not independent training repetitions. One seed and one final candidate were used. Differences must not be presented as clinical, lesion-specific, scanner acquisition-time or cost benefits. The original VAE was trained under historical preprocessing; its weights remain frozen despite corrected acquired-only preprocessing, so residual domain mismatch is possible. No CaLID/ISCS/commercial-product head-to-head experiment was conducted. Historical results have different preprocessing and are kept separate.

## Verification

All 34 tests passed. Training smoke verification ran two steps and then resumed to cumulative step three. End-to-end evaluation repeatedly asserts exact preservation of acquired planes and exports a sample research NIfTI with unknown physical geometry. Both one-channel and four-channel NIfTI/HDF5 exports were checked. Full dataset payloads were sampled, while all filenames/indices and metadata were audited.

`reproducibility_check.json` verifies an exact repeated reconstruction of volume 14 at stride 8 with seed 1450: maximum absolute difference 0, and acquired planes exactly preserved. This rerun used the final reviewed source.

The disclosure addresses all 29 sections, preserves the supplied outer/nested table structure and embeds four schematics. Microsoft Word's COM server failed to initialize (80080005), so native page rendering could not be verified here. Structural checks and direct drawing inspection are recorded; inspect final pagination in Word before using the form.
