"""Regression tests for the substantive review corrections."""
import types
import numpy as np
import torch
import pytest
import h5py
import nibabel as nib
from src.data.preprocessing import normalize_acquired_volume, validate_acquired
from src.data.dataset import BraTSSliceDataset, acquired_slice_stats, _load_slice_from_h5, get_patient_splits
from src.data.sparsifier import VolumeSparsifier
from src.models.pipeline import SparseToDenseDiffusionPipeline
from src.models.scheduler import DiffusionNoiseScheduler
from src.models.factory import load_checkpoint, build_pipeline, ROOT
from src.utils.volume_io import export_volume

def test_normalization_does_not_read_missing_intensities():
    v = torch.arange(1, 10, dtype=torch.float32).view(9,1,1,1).expand(9,1,4,4).clone()
    a = [0,4,8]
    x, stats = normalize_acquired_volume(v, a)
    v[[1,2,3,5,6,7]] = 1e8
    y, changed = normalize_acquired_volume(v, a)
    assert stats == changed
    assert torch.equal(x[a], y[a])

def test_missing_targets_and_shared_stats(tmp_path):
    for z in range(9):
        with h5py.File(tmp_path / f'volume_1_slice_{z}.h5', 'w') as f:
            f['image'] = np.full((8,8,4), z+1, dtype=np.float32)
            f['mask'] = np.zeros((8,8,3), dtype=np.uint8)
    ds = BraTSSliceDataset(str(tmp_path), [1], stride=4, total_slices_per_vol=9)
    assert {s['target_z'] for s in ds.samples} == {1,2,3,5,6,7}
    sample = ds[0]
    assert sample['stats_max'] < 10
    assert sample['prev'].shape == (1,8,8)

def fake_pipeline():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.in_channels = 1
    p = SparseToDenseDiffusionPipeline(Dummy(), Dummy(), Dummy(), DiffusionNoiseScheduler())
    def generate(self, prev_slices, next_slices, normalized_zs, alphas, gap_dists, **kw):
        a = alphas[:,None,None,None]
        return (1-a)*prev_slices+a*next_slices
    p.generate_slices_batch = types.MethodType(generate, p)
    return p

@pytest.mark.parametrize('smooth', [True,False])
def test_irregular_indices_and_exact_acquired(smooth):
    p = fake_pipeline()
    vol = torch.zeros(9,1,8,8)
    acquired = [8,0,3,7]
    for z in acquired:
        vol[z] = z
    result = p.reconstruct_volume(vol, acquired, stride=4, apply_smoothing=smooth)
    assert torch.equal(result[acquired], vol[acquired])
    assert result[4].mean() > 0
    if not smooth:
        assert torch.allclose(result[6], torch.full_like(result[6], 6))

@pytest.mark.parametrize('indices', [[],[0,0,8],[-1,8],[0,9],[0,1.5,8]])
def test_invalid_indices(indices):
    with pytest.raises(ValueError):
        validate_acquired(indices, 9)

def test_checkpoint_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / 'missing.pt')
    torch.save({}, tmp_path / 'bad.pt')
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / 'bad.pt')

@pytest.mark.parametrize('channels', [1,4])
def test_exports(tmp_path, channels):
    v = np.arange(9*channels*8*8,dtype=np.float32).reshape(9,channels,8,8)
    export_volume(v, tmp_path / 'volume.nii.gz')
    nii = nib.load(tmp_path / 'volume.nii.gz')
    expected = v[:,0].transpose(1,2,0) if channels == 1 else v.transpose(2,3,0,1)
    np.testing.assert_array_equal(nii.get_fdata(), expected)
    assert int(nii.header['sform_code']) == 0
    export_volume(v, tmp_path / 'volume.h5')
    with h5py.File(tmp_path / 'volume.h5') as f:
        np.testing.assert_array_equal(f['reconstructed'][()], v)

def test_split_rng_and_repeatability(tmp_path):
    for i in range(20):
        (tmp_path / f'volume_{i}_slice_0.h5').touch()
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    splits = get_patient_splits(str(tmp_path))
    assert np.array_equal(before,np.random.get_state()[1])
    assert splits == get_patient_splits(str(tmp_path))
    assert not set(splits[0]) & set(splits[1])
    assert not set(splits[0]) & set(splits[2])

def test_scheduler_seed_and_limits():
    s = DiffusionNoiseScheduler()
    x = torch.zeros(1,4,4,4)
    a = s.ddim_step(x,500,400,x,eta=1,generator=torch.Generator().manual_seed(42))[0]
    b = s.ddim_step(x,500,400,x,eta=1,generator=torch.Generator().manual_seed(42))[0]
    assert torch.equal(a,b)
    for steps in [0,-1,1001]:
        with pytest.raises(ValueError):
            s.get_ddim_timesteps(steps)


def test_selection_requires_psnr_gain_without_ssim_loss():
    import pandas as pd
    from scripts.review_experiment import select
    ref = pd.DataFrame([dict(weights='raw',steps=25,smoothing=True,psnr_missing=27.,ssim_missing=.90)])
    candidate = pd.DataFrame([
        dict(weights='raw',steps=25,smoothing=True,psnr_missing=28.,ssim_missing=.89),
        dict(weights='ema',steps=50,smoothing=True,psnr_missing=27.1,ssim_missing=.90)])
    selection = select(ref,candidate)
    assert selection['use_candidate'] and selection['weights']=='ema'
    assert not select(ref,candidate.iloc[:1])['use_candidate']


def test_nifti_viewer_uses_explicit_export_axes(tmp_path):
    from scripts.view_volume import load_nifti
    volume=np.arange(12*4*8*8,dtype=np.float32).reshape(12,4,8,8)
    export_volume(volume,tmp_path/'multimodal.nii.gz')
    np.testing.assert_array_equal(load_nifti(str(tmp_path/'multimodal.nii.gz'),channel=2),volume[:,2])


def test_complete_acquisition_still_validates_sampling():
    with pytest.raises(ValueError):
        fake_pipeline().reconstruct_volume(torch.zeros(9,1,8,8),list(range(9)),num_inference_steps=0)
