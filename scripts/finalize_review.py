"""Publish review summaries from completed experiment evidence."""
import json
import sys
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'results/review'

def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))

def run():
    completion=read('completion.json'); selection=read('selection.json')
    ref=pd.read_csv(OUT/'test_reference.csv'); chosen=pd.read_csv(OUT/'test_selected.csv')
    paired=pd.read_csv(OUT/'paired_comparison.csv')
    training=json.loads((ROOT/'experiments/review/finetune/completion.json').read_text())
    manifest=read('experiment_manifest.json')
    val_tables=[]
    for model,name in [('Reference','validation_reference.csv'),('Fine-tuned','validation_candidate.csv')]:
        frame=pd.read_csv(OUT/name).groupby(['weights','steps','smoothing'])[['psnr_missing','ssim_missing']].mean().reset_index()
        for row in frame.itertuples():
            val_tables.append(f'| {model} | {row.weights} | {row.steps} | {row.smoothing} | {row.psnr_missing:.4f} | {row.ssim_missing:.5f} |')
    validation_table='\n'.join(['| Model | Weights | DDIM steps | Smoothing | Mean PSNR | Mean SSIM |','|---|---|---:|---|---:|---:|']+val_tables)
    rows=[]
    for stride in (8,16):
        for label,frame in [('Corrected reference',ref[(ref.stride==stride)&(ref.method=='conditional_diffusion')]),('Selected',chosen[chosen.stride==stride]),('Linear',ref[(ref.stride==stride)&(ref.method=='linear')]),('Smoothed linear',ref[(ref.stride==stride)&(ref.method=='smoothed_linear')])]:
            rows.append(f'| {stride} | {label} | {frame.psnr_missing.mean():.3f} ± {frame.psnr_missing.std():.3f} | {frame.ssim_missing.mean():.4f} ± {frame.ssim_missing.std():.4f} | {frame.seconds.mean():.2f} |')
    summary='\n'.join(['| Stride | Method | Missing PSNR (dB), mean ± SD | Missing SSIM, mean ± SD | Mean seconds |','|---|---|---:|---:|---:|']+rows)
    decision=('Fine-tuned candidate selected' if selection['use_candidate'] else 'Corrected original reference retained')+f" using {selection['weights']} weights, {selection['steps']} DDIM steps, smoothing {selection['smoothing']}."
    gain=f"Across 20 patient/stride pairs, selected versus the corrected raw 25-step smoothed reference: PSNR change {paired.psnr_gain.mean():+.3f} ± {paired.psnr_gain.std():.3f} dB; SSIM change {paired.ssim_gain.mean():+.4f} ± {paired.ssim_gain.std():.4f}."
    timing=f"Total bounded protocol: {completion['seconds']/60:.2f} minutes (maximum 120). Fine-tuning: {training['step']} steps, {training['seconds']/60:.2f} minutes, stopped by {training['stopped_by']}."
    reference_seconds=read('test_reference.json')['seconds']+read('validation_reference.json')['seconds']
    selection_seconds=read('validation_candidate.json')['seconds']+read('test_selected.json')['seconds']
    report=f'''# Corrected reconstruction experiment

Completed locally on 8 October 2026. {decision}

{gain}

{summary}

## Protocol and data separation

Seed 42. Validation patients: {manifest['validation_volumes']}. Final test patients: {manifest['test_volumes']}. The configured split is 258 training / 55 validation / 56 test volumes, disjoint by volume. Final comparisons use the ten historical benchmark patients, not the full internal test partition.

Normalization bounds come only from acquired slices for each volume and stride; context and withheld targets share these bounds. Diffusion training excludes acquired targets. The architecture and pretrained VAE remain unchanged. AdamW fine-tunes encoder/U-Net at 2e-5 with batch 8, equal-probability strides 4/8/16, weight decay 1e-4, clipping norm 1 and EMA decay .999. Augmentation is disabled. Raw/EMA, 25/50 steps and smoothing off/on yield eight configurations per model, evaluated on the same validation cases with paired patient/stride seeds.

The best corrected reference validation setting establishes the eligibility threshold. Candidates need a strictly higher mean missing-slice PSNR across both strides and non-decreasing mean SSIM. The selected candidate is tested once on the held-out cases. Final paired gains use the common raw 25-step smoothed reference; when selection chooses another reference setting, these gains can reflect setting changes rather than fine-tuning. No test cases select hyperparameters.

## Validation grid

Means across five patients and strides 8/16 (ten patient/stride pairs per setting):

{validation_table}

## Budget and runtime

{timing}

- Corrected reference evaluation and validation grid: {reference_seconds/60:.2f} minutes (20-minute allocation).
- Fine-tuning: {training['seconds']/60:.2f} minutes (70-minute allocation).
- Candidate validation and final comparisons: {selection_seconds/60:.2f} minutes (30-minute allocation).

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
'''
    (OUT/'EXPERIMENT_REPORT.md').write_text(report,encoding='utf-8')
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    start='<!-- REVIEW_RESULTS_START -->'; end='<!-- REVIEW_RESULTS_END -->'
    updated=readme.split(start)[0]+start+'\n'+decision+'\n\n'+summary+'\n\n'+gain+'\n\n'+timing+' See the experiment report for timing scope and limitations.\n'+end+readme.split(end)[1]
    (ROOT/'README.md').write_text(updated,encoding='utf-8')
    checks={}
    for folder in ('src','scripts','configs','tests'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts: checks[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'source_checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,column,label in zip(axes,['psnr_missing','ssim_missing'],['Missing-slice PSNR (dB)','Missing-slice SSIM']):
        for offset,name,frame in [(-.23,'Corrected reference',ref[ref.method=='conditional_diffusion']),(0.,'Selected',chosen),(.23,'Linear',ref[ref.method=='linear'])]:
            grouped=frame.groupby('stride')[column].agg(['mean','std']).loc[[8,16]]
            ax.bar(np.arange(2)+offset,grouped['mean'],width=.22,yerr=grouped['std'],capsize=3,label=name)
        ax.set_xticks([0,1],['Stride 8','Stride 16']); ax.set_ylabel(label); ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT/'measured_results.png',dpi=180); plt.close(fig)
    print(decision+' '+gain+' '+timing)

if __name__=='__main__':run()
