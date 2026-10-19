"""Corrected benchmark CLI; historical results are never overwritten by default."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from src.models.factory import ROOT, read_configs, build_pipeline, seed_everything, checkpoint_id
from src.data.dataset import BraTSVolumeDataset, configured_patient_splits
from src.evaluation.evaluator import VolumeBenchmarkEvaluator
import json
import yaml

def run_benchmark(max_test_vols=5, strides=None, use_diffusion=True,
    diffusion_ckpt='experiments/checkpoints/diffusion/best_model.pt', vae_ckpt='experiments/checkpoints/vae/best_model.pt',
    output_dir='results/review/benchmark', seed=42, weights='raw'):
    if max_test_vols < 1:
        raise ValueError('At least one test volume is required')
    seed_everything(seed)
    cfg=read_configs()[0]
    evaluation=yaml.safe_load((ROOT/'configs/eval_config.yaml').read_text(encoding='utf-8'))['evaluation']
    strides=evaluation['strides'] if strides is None else strides
    _,_,ids=configured_patient_splits(cfg,seed=seed)
    ds=BraTSVolumeDataset(str(ROOT/cfg['data']['data_dir']),ids[:max_test_vols],cfg['data']['modality'],cfg['data']['total_slices'])
    pipeline=build_pipeline(ROOT/vae_ckpt,ROOT/diffusion_ckpt,weights=weights) if use_diffusion else None
    device=next(pipeline.parameters()).device if pipeline is not None else torch.device('cpu')
    with torch.autocast(device.type,enabled=device.type=='cuda'):
        frame=VolumeBenchmarkEvaluator(ds,list(strides),baselines=evaluation['baselines'],diffusion_pipeline=pipeline,device=device).evaluate_all()
    out=ROOT/output_dir
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'metrics.csv',index=False)
    summary=frame.groupby(['stride','method'])[['psnr_missing','ssim_missing','mae_missing']].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    (out/'manifest.json').write_text(json.dumps({'seed':seed,'test_volumes':ds.volume_ids,'strides':list(strides),
        'normalization':'acquired-only','weights':weights,'configs':read_configs(),
        'checkpoint':checkpoint_id(ROOT/diffusion_ckpt) if use_diffusion else None},indent=2))
    print(summary)
    return frame

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vols',type=int,default=3)
    p.add_argument('--no_diff',action='store_true')
    p.add_argument('--strides',type=int,nargs='+',help='Override configured strides')
    p.add_argument('--output-dir',default='results/review/benchmark')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--weights',choices=['raw','ema'],default='raw')
    p.add_argument('--diffusion-ckpt',default='experiments/checkpoints/diffusion/best_model.pt')
    a=p.parse_args()
    run_benchmark(a.vols,a.strides,not a.no_diff,a.diffusion_ckpt,output_dir=a.output_dir,seed=a.seed,weights=a.weights)

if __name__=='__main__':
    main()
