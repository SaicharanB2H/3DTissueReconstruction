"""
Demo Visualization Script: Renders 2D Orthogonal Comparison Grids and 3D Volume Visualizations.
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import BraTSVolumeDataset, get_patient_splits
from src.data.sparsifier import VolumeSparsifier
from src.baselines.interpolators import get_baseline_interpolator
from src.visualization.slice_viewer import SliceComparisonViewer
from src.visualization.volume_viewer_3d import Volume3DVisualizer
from src.utils.logger import setup_logger


def generate_visual_demos(
    volume_id: int = 100,
    stride: int = 4,
    slice_indices: list = [50, 75, 100],
    output_dir_2d: str = "./results/visualizations",
    output_dir_3d: str = "./results/visualizations_3d",
):
    with open("configs/default_config.yaml", "r") as f:
        default_cfg = yaml.safe_load(f)

    device = "cpu"
    logger = setup_logger("DemoVisualize")
    data_dir = default_cfg["data"]["data_dir"]
    modality = default_cfg["data"]["modality"]

    # 1. Load dataset & Volume
    dataset = BraTSVolumeDataset(data_dir=data_dir, volume_ids=[volume_id], modality=modality)
    item = dataset[0]
    gt_vol = item["volume"] # (D, C, H, W)
    total_slices = dataset.total_slices

    # 2. Sparsify
    sparsifier = VolumeSparsifier(stride=stride, total_slices=total_slices)
    sparse_vol, mask, acquired, missing = sparsifier.sparsify_volume(gt_vol, stride=stride)

    # 3. Generate baseline reconstructions
    predictions_by_method = {}
    for baseline in ["nearest", "linear", "cubic", "bspline_3d"]:
        interp = get_baseline_interpolator(baseline)
        recon = interp.reconstruct(sparse_vol, acquired, total_slices)
        predictions_by_method[baseline.capitalize()] = recon

    # 4. Generate 2D Multi-panel comparison slices
    slice_viewer = SliceComparisonViewer(output_dir=output_dir_2d)
    for z in slice_indices:
        pred_slices = {m: predictions_by_method[m][z] for m in predictions_by_method}
        p_2d = slice_viewer.plot_slice_grid(
            gt_slice=gt_vol[z],
            predictions=pred_slices,
            slice_idx=z,
            volume_id=volume_id,
            stride=stride,
        )
        logger.info(f"Generated 2D slice comparison at z={z}: {p_2d}")

    # 5. Generate Orthogonal views
    linear_vol = predictions_by_method["Linear"]
    ortho_path = slice_viewer.plot_orthogonal_views(
        gt_volume=gt_vol,
        recon_volume=linear_vol,
        method_name="Linear Baseline",
        volume_id=volume_id,
        stride=stride,
    )
    logger.info(f"Generated Orthogonal views: {ortho_path}")

    # 6. Generate 3D PyVista volume renders
    try:
        vol_3d_viewer = Volume3DVisualizer(output_dir=output_dir_3d)
        render_3d_path = vol_3d_viewer.render_sparse_vs_dense_3d(
            sparse_volume=sparse_vol,
            dense_volume=linear_vol,
            gt_volume=gt_vol,
            volume_id=volume_id,
            stride=stride,
        )
        logger.info(f"Generated 3D volume rendering: {render_3d_path}")

        ortho_3d_path = vol_3d_viewer.render_ortho_slices_3d(
            volume=linear_vol,
            title=f"3D Orthogonal Planes (Vol {volume_id})",
        )
        logger.info(f"Generated 3D orthogonal planes: {ortho_3d_path}")
    except Exception as e:
        logger.warning(f"3D rendering skipped: {e}")

    logger.info("Visual demos generation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol_id", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    generate_visual_demos(volume_id=args.vol_id, stride=args.stride)
