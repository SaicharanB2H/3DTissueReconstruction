"""
Interactive and Multi-Planar Viewer for Reconstructed Medical NIfTI (.nii.gz) Volumes.
Supports:
1. Interactive slice-by-slice navigation with a slider or keyboard arrow keys.
2. Side-by-side comparison with ground-truth volume (with error heatmap).
3. Exporting key multi-slice grid images for reports and presentations.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_nifti(path: str, channel: int = 0) -> np.ndarray:
    """Load project exports from H,W,D[,C] into D,H,W; axes are not patient orientation."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    # Squeeze channel dim if present
    if data.ndim == 4:
        data = data[..., channel]
    if data.ndim != 3:
        raise ValueError('Expected H,W,D or H,W,D,C project export')
    return np.transpose(data, (2, 0, 1))


def interactive_viewer(
    recon_vol: np.ndarray,
    gt_vol: np.ndarray = None,
    title: str = "Reconstructed 3D Volume",
):
    """
    Launches an interactive Matplotlib window with an axial slice slider.
    """
    D, H, W = recon_vol.shape
    init_z = D // 2

    has_gt = gt_vol is not None and gt_vol.shape == recon_vol.shape

    if has_gt:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{title} (Slice {init_z}/{D-1})", fontsize=14, fontweight="bold")
        plt.subplots_adjust(bottom=0.2)

        im_recon = axes[0].imshow(recon_vol[init_z], cmap="gray", origin="lower", vmin=-1, vmax=1)
        axes[0].set_title("Reconstructed")
        axes[0].axis("off")

        im_gt = axes[1].imshow(gt_vol[init_z], cmap="gray", origin="lower", vmin=-1, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        err = np.abs(recon_vol[init_z] - gt_vol[init_z])
        im_err = axes[2].imshow(err, cmap="hot", origin="lower", vmin=0, vmax=0.2)
        axes[2].set_title("Absolute Error (|Recon - GT|)")
        axes[2].axis("off")
        fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)

    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.suptitle(f"{title} (Slice {init_z}/{D-1})", fontsize=14, fontweight="bold")
        plt.subplots_adjust(bottom=0.2)

        im_recon = ax.imshow(recon_vol[init_z], cmap="gray", origin="lower", vmin=-1, vmax=1)
        ax.set_title("Axial Slice")
        ax.axis("off")

    # Slider
    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.04], facecolor="lightgray")
    slider = Slider(ax_slider, "Slice Z", 0, D - 1, valinit=init_z, valstep=1, valfmt="%d")

    def update(val):
        z = int(slider.val)
        fig.suptitle(f"{title} (Slice {z}/{D-1})", fontsize=14, fontweight="bold")
        im_recon.set_data(recon_vol[z])
        if has_gt:
            im_gt.set_data(gt_vol[z])
            im_err.set_data(np.abs(recon_vol[z] - gt_vol[z]))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Keyboard navigation (Left/Down: previous slice, Right/Up: next slice)
    def on_key(event):
        curr = int(slider.val)
        if event.key in ["right", "up"] and curr < D - 1:
            slider.set_val(curr + 1)
        elif event.key in ["left", "down"] and curr > 0:
            slider.set_val(curr - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\n" + "=" * 60)
    print(" Interactive Viewer Controls:")
    print(" - Drag the slider at the bottom")
    print(" - Press Left / Down arrow keys for previous slice")
    print(" - Press Right / Up arrow keys for next slice")
    print(" - Close the window to exit")
    print("=" * 60 + "\n")

    plt.show()


def export_slice_montage(
    recon_vol: np.ndarray,
    out_path: str = "./results/visualizations/recon_montage.png",
    num_slices: int = 12,
):
    """Exports a multi-slice grid montage of the volume."""
    D, H, W = recon_vol.shape
    step = D // (num_slices + 1)
    slice_indices = [step * (i + 1) for i in range(num_slices)]

    rows = int(np.ceil(num_slices / 4))
    cols = min(num_slices, 4)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for i, z in enumerate(slice_indices):
        axes[i].imshow(recon_vol[z], cmap="gray", origin="lower", vmin=-1, vmax=1)
        axes[i].set_title(f"Slice z = {z}", fontsize=11)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved slice montage to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="View Reconstructed NIfTI Volume")
    parser.add_argument(
        "--file",
        type=str,
        default="./results/reconstructed_volumes/recon_vol_100_stride_4x.nii.gz",
        help="Path to .nii.gz volume",
    )
    parser.add_argument(
        "--vol_id",
        type=int,
        default=None,
        help="Volume ID to automatically find ground truth from dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./BraTS2020_training_data/content/data",
        help="Data directory for ground truth lookup",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        help="Export a 12-slice montage PNG image",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    print(f"Loading reconstructed volume: {args.file}")
    recon_vol = load_nifti(args.file)
    print(f"Volume shape: {recon_vol.shape} | Intensity range: [{recon_vol.min():.3f}, {recon_vol.max():.3f}]")

    # Try loading ground truth if volume_id is given or discoverable
    gt_vol = None
    vol_id = args.vol_id
    if vol_id is None:
        base = os.path.basename(args.file)
        if "vol_" in base:
            try:
                vol_id = int(base.split("vol_")[1].split("_")[0])
            except Exception:
                pass

    if vol_id is not None:
        from src.data.dataset import BraTSVolumeDataset
        try:
            ds = BraTSVolumeDataset(args.data_dir, [vol_id], modality="flair")
            gt_data = ds[0]
            sidecar = Path(args.file + '.json')
            if not sidecar.exists():
                raise ValueError('Normalization sidecar required for a comparable ground-truth view')
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            from src.data.transforms import MedicalIntensityNormalizer
            gt_vol, _ = MedicalIntensityNormalizer().normalize(gt_data['volume_raw'], stats=metadata['normalization_stats'])
            gt_vol = gt_vol.numpy()[:, 0]
            print(f"Loaded ground truth for Volume {vol_id}")
        except Exception as e:
            print(f"Could not load ground truth: {e}")

    # Always export montage if requested or as a static preview
    montage_path = args.file.replace(".nii.gz", "_montage.png")
    export_slice_montage(recon_vol, out_path=montage_path)

    # Launch interactive viewer
    if not args.montage:
        interactive_viewer(recon_vol, gt_vol=gt_vol, title=f"Reconstruction (Vol {vol_id or 'Unknown'})")


if __name__ == "__main__":
    main()
