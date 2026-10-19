"""Reproducible reference, fine-tuning and validation selection within local time budgets."""
import argparse
import csv
import json
import sys
import time
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from src.models.factory import ROOT, read_configs, build_pipeline, checkpoint_id, seed_everything
from src.data.dataset import BraTSVolumeDataset, get_patient_splits
from src.data.preprocessing import normalize_acquired_volume
from src.data.sparsifier import VolumeSparsifier
from src.baselines.interpolators import get_baseline_interpolator
from src.evaluation.metrics import evaluate_volume_reconstruction
from src.utils.volume_io import export_volume
from scripts.finetune_diffusion import train, parser as training_parser

OUT = ROOT / 'results/review'
REFERENCE = ROOT / 'experiments/checkpoints/diffusion/best_model.pt'
VAE = ROOT / 'experiments/checkpoints/vae/best_model.pt'

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def evaluate(checkpoint, volumes, split, weights_options=('raw',), steps_options=(25,),
             smoothing_options=(True,), strides=(8,16), baselines=False, deadline=None, seed=42, output=None):
    cfg = read_configs()[0]
    output = output or OUT / f'{split}.csv'
    dataset = BraTSVolumeDataset(str(ROOT / cfg['data']['data_dir']), volumes, cfg['data']['modality'], cfg['data']['total_slices'])
    records = []
    started = time.monotonic()
    identity = checkpoint_id(checkpoint)
    for weight in weights_options:
        pipeline = build_pipeline(VAE, checkpoint, weights=weight)
        device = next(pipeline.parameters()).device
        for i in range(len(dataset)):
            raw = dataset[i]['volume_raw']
            for stride in strides:
                sp = VolumeSparsifier(stride, len(raw))
                acquired, missing = sp.get_acquired_indices(), sp.get_missing_indices()
                gt, stats = normalize_acquired_volume(raw, acquired)
                gt = gt.to(device)
                sparse, _, _, _ = sp.sparsify_volume(gt)
                if baselines and weight == weights_options[0]:
                    for method in ('nearest','linear','cubic','smoothed_linear'):
                        t0 = time.monotonic()
                        recon = get_baseline_interpolator(method).reconstruct(sparse, acquired, len(raw))
                        m = evaluate_volume_reconstruction(recon,gt,missing)
                        records.append(dict(m, vol_id=volumes[i], stride=stride, method=method, weights='', steps=0, smoothing=False,
                                            seconds=time.monotonic()-t0, seed=seed, split=split, checkpoint_sha256=''))
                for steps in steps_options:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(f'Evaluation budget exhausted; completed {len(records)} rows')
                    # Same seed for a patient/stride across all candidates, independent of candidate order.
                    sample_seed = seed + volumes[i]*100 + stride
                    t0 = time.monotonic()
                    with torch.autocast(device.type, enabled=device.type == 'cuda'):
                        generated = pipeline.reconstruct_volume(sparse, acquired, stride, num_inference_steps=steps,
                            apply_smoothing=False, generator=torch.Generator(device=device).manual_seed(sample_seed))
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    generation_seconds = time.monotonic()-t0
                    for smooth in smoothing_options:
                        if smooth:
                            arr = generated.float().cpu().numpy()
                            arr_s = gaussian_filter(arr, sigma=(.5,0,0,0))
                            arr_s[acquired] = arr[acquired]
                            recon = torch.from_numpy(arr_s).to(device)
                        else:
                            recon = generated
                        if not torch.equal(recon[acquired], sparse[acquired]):
                            raise AssertionError('Acquired slice changed')
                        m = evaluate_volume_reconstruction(recon, gt, missing)
                        row = dict(m, vol_id=volumes[i], stride=stride, method='conditional_diffusion', weights=weight,
                            steps=steps, smoothing=smooth, seconds=generation_seconds, seed=sample_seed,
                            split=split, checkpoint_sha256=identity['sha256'])
                        records.append(row)
                        pd.DataFrame(records).to_csv(output, index=False)
                        print(f'{split} weight={weight} vol={volumes[i]} stride={stride} steps={steps} smoothing={smooth} PSNR={m["psnr_missing"]:.3f} SSIM={m["ssim_missing"]:.4f} seconds={generation_seconds:.1f}', flush=True)
                        if split == 'test_selected' and i == 0 and stride == strides[0]:
                            export_volume(recon, OUT / 'example_reconstruction.nii.gz', {'normalization_stats': stats,
                                'volume_id': volumes[i], 'stride':stride, 'checkpoint': identity, 'metrics':m,
                                'seed':sample_seed,'weights':weight,'steps':steps,'smoothing':smooth,
                                'acquired_indices':acquired,'vae_checkpoint':checkpoint_id(VAE)})
                            np.savez_compressed(OUT / 'example_comparison.npz', gt=gt.cpu().numpy(), recon=recon.cpu().numpy(), sparse=sparse.cpu().numpy())
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    group = ['stride','method','weights','steps','smoothing']
    frame.groupby(group, dropna=False)[['psnr_missing','ssim_missing','mae_missing','seconds']].agg(['mean','std']).to_csv(output.with_name(output.stem+'_summary.csv'))
    write_json(output.with_suffix('.json'), {'checkpoint': identity, 'vae_checkpoint':checkpoint_id(VAE),
        'volumes':volumes, 'split':split, 'strides':list(strides), 'weights':list(weights_options),
        'steps':list(steps_options), 'smoothing':list(smoothing_options), 'seed':seed,
        'normalization':'acquired-only per volume/stride', 'configs':read_configs(),
        'seconds':time.monotonic()-started, 'torch_version':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'dtype':'CUDA autocast float16; float32 scheduler and metrics', 'batch_size':8})
    return frame

def select(reference, candidate):
    keys = ['weights','steps','smoothing']
    reference_scores = reference.groupby(keys)[['psnr_missing','ssim_missing']].mean().reset_index()
    reference_best = reference_scores.sort_values(['psnr_missing','ssim_missing'], ascending=False).iloc[0]
    scores = candidate.groupby(keys)[['psnr_missing','ssim_missing']].mean().reset_index()
    eligible = scores[(scores.psnr_missing > reference_best.psnr_missing) & (scores.ssim_missing >= reference_best.ssim_missing)]
    use_candidate = not eligible.empty
    best = eligible.sort_values(['psnr_missing','ssim_missing'], ascending=False).iloc[0] if use_candidate else reference_best
    return {'use_candidate': use_candidate, 'weights':str(best['weights']), 'steps':int(best['steps']),
        'smoothing':bool(best['smoothing']), 'validation_psnr_missing':float(best['psnr_missing']),
        'validation_ssim_missing':float(best['ssim_missing']), 'reference_best':reference_best.to_dict(),
        'criterion':'mean missing PSNR across five validation patients and strides 8/16, non-decreasing SSIM'}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    seed_everything(42)
    torch.set_num_threads(4)
    cfg = read_configs()[0]
    from src.data.dataset import configured_patient_splits
    train_vols,val_vols,test_vols = configured_patient_splits(cfg, seed=42)
    historical = sorted(pd.read_csv(ROOT/'results/benchmark_metrics.csv').vol_id.unique().tolist())
    assert set(historical) <= set(test_vols)
    validation = val_vols[:5]
    started = time.monotonic()
    final_deadline = started + 120*60
    write_json(OUT/'experiment_manifest.json', {'validation_volumes':validation,'test_volumes':historical,
        'train_count':len(train_vols),'seed':42,'total_budget_seconds':7200,'stage_budgets_seconds':[1200,4200,1800]})
    reference_deadline = started + 1200
    ref_test = evaluate(REFERENCE,historical,'test_reference',baselines=True,deadline=reference_deadline)
    ref_validation = evaluate(REFERENCE,validation,'validation_reference',('raw','ema'),(25,50),(False,True),deadline=reference_deadline)
    training_args = training_parser().parse_args([])
    candidate_path = Path(train(training_args))
    selection_started = time.monotonic()
    selection_deadline = min(selection_started+1800,final_deadline)
    candidate_validation = evaluate(candidate_path,validation,'validation_candidate',('raw','ema'),(25,50),(False,True),deadline=selection_deadline)
    chosen = select(ref_validation,candidate_validation)
    chosen['checkpoint'] = str(candidate_path if chosen['use_candidate'] else REFERENCE)
    write_json(OUT/'selection.json',chosen)
    selected = evaluate(Path(chosen['checkpoint']),historical,'test_selected',(chosen['weights'],),
        (chosen['steps'],),(chosen['smoothing'],),deadline=selection_deadline)
    # Compare both against linear and a common corrected 25-step raw reference.
    paired = selected.merge(ref_test[ref_test.method=='conditional_diffusion'],on=['vol_id','stride'],suffixes=('_selected','_reference'))
    paired['psnr_gain']=paired.psnr_missing_selected-paired.psnr_missing_reference
    paired['ssim_gain']=paired.ssim_missing_selected-paired.ssim_missing_reference
    paired.to_csv(OUT/'paired_comparison.csv',index=False)
    write_json(OUT/'completion.json', {'status':'completed','seconds':time.monotonic()-started,
        'use_candidate':chosen['use_candidate'],'mean_test_psnr_gain':float(paired.psnr_gain.mean()),
        'mean_test_ssim_gain':float(paired.ssim_gain.mean()),'reference_validation_settings':chosen['reference_best']})

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        write_json(OUT/'failure.json',{'error':repr(e),'status':'failed; partial outputs retained'})
        raise
