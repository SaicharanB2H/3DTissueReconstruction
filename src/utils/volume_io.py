"""Research exports with explicit axis and geometry provenance."""
from pathlib import Path
import json
import numpy as np
import h5py
import nibabel as nib

def export_volume(volume, path, metadata=None, affine=None):
    arr = volume.detach().cpu().numpy() if hasattr(volume, 'detach') else np.asarray(volume)
    if arr.ndim != 4 or not np.isfinite(arr).all():
        raise ValueError('Expected finite (D,C,H,W) volume')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    info = dict(metadata or {})
    info.update({'input_axes': 'D,C,H,W', 'intensity_space': 'normalized [-1,1]',
        'geometry': 'provided affine' if affine is not None else 'unknown; identity affine uses voxel indices, not verified patient coordinates'})
    if p.name.endswith(('.nii', '.nii.gz')):
        data = arr[:, 0].transpose(1, 2, 0) if arr.shape[1] == 1 else arr.transpose(2, 3, 0, 1)
        img = nib.Nifti1Image(data.astype(np.float32), np.eye(4) if affine is None else affine)
        if affine is None:
            img.set_qform(np.eye(4), code=0)
            img.set_sform(np.eye(4), code=0)
            img.header.set_xyzt_units('unknown')
        img.header['descrip'] = b'Research reconstruction; geometry unverified' if affine is None else b'Research reconstruction'
        nib.save(img, p)
        info['output_axes'] = 'H,W,D' if arr.shape[1] == 1 else 'H,W,D,C'
    elif p.suffix in ('.h5', '.hdf5'):
        with h5py.File(p, 'w') as f:
            f.create_dataset('reconstructed', data=arr, compression='gzip')
            f.attrs['metadata'] = json.dumps(info)
        info['output_axes'] = 'D,C,H,W'
    else:
        raise ValueError('Supported exports: .nii, .nii.gz, .h5, .hdf5')
    p.with_name(p.name + '.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return str(p)
