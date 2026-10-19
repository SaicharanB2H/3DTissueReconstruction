"""Check seeded reconstruction against the saved selected example and verify preservation."""
import sys
import json
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
import pandas as pd
from src.models.factory import build_pipeline,seed_everything,checkpoint_id
from src.data.sparsifier import VolumeSparsifier
OUT=ROOT/'results/review'

def run():
    seed_everything(42);torch.set_num_threads(4)
    chosen=json.loads((OUT/'selection.json').read_text())
    sidecar=OUT/'example_reconstruction.nii.gz.json'
    info=json.loads(sidecar.read_text())
    frame=pd.read_csv(OUT/'test_selected.csv')
    row=frame[(frame.vol_id==info['volume_id'])&(frame.stride==info['stride'])].iloc[0]
    acquired=VolumeSparsifier(int(row.stride),155).get_acquired_indices()
    info.update({'seed':int(row.seed),'weights':str(row.weights),'steps':int(row.steps),'smoothing':bool(row.smoothing),'acquired_indices':acquired,'vae_checkpoint':checkpoint_id(ROOT/'experiments/checkpoints/vae/best_model.pt')})
    sidecar.write_text(json.dumps(info,indent=2),encoding='utf-8')
    pipeline=build_pipeline(ROOT/'experiments/checkpoints/vae/best_model.pt',chosen['checkpoint'],weights=chosen['weights'])
    device=next(pipeline.parameters()).device
    saved=np.load(OUT/'example_comparison.npz')
    sparse=torch.from_numpy(saved['sparse']).to(device)
    target=torch.from_numpy(saved['recon']).to(device)
    started=time.monotonic()
    with torch.autocast(device.type,enabled=device.type=='cuda'):
        repeated=pipeline.reconstruct_volume(sparse,acquired,int(row.stride),num_inference_steps=int(row.steps),apply_smoothing=bool(row.smoothing),generator=torch.Generator(device=device).manual_seed(int(row.seed)))
    assert torch.equal(repeated[acquired],sparse[acquired])
    maximum=float((repeated-target).abs().max())
    assert torch.equal(repeated,target),'Repeated reconstruction differs from the saved seeded example'
    result={'status':'passed','exact_repeat_of_saved_selected_example':True,'max_absolute_difference':maximum,'acquired_slices_preserved_exactly':True,'volume_id':info['volume_id'],'stride':int(row.stride),'seed':int(row.seed),'steps':int(row.steps),'weights':chosen['weights'],'seconds':time.monotonic()-started,'checkpoint':checkpoint_id(chosen['checkpoint'])}
    (OUT/'reproducibility_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
