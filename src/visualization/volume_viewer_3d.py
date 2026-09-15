"""
3D Medical Volume Rendering Module using PyVista and Open3D.
Generates 3D ray-cast volume views, orthogonal slicing planes, and isosurfaces.
"""
from typing import Optional, Union, Tuple
import os
import numpy as np
import torch
import pyvista as pv


class Volume3DVisualizer:
    """
    Renders 3D brain volumes, sparse acquisition stacks with gaps,
    and reconstructed dense volumes using PyVista.
    """

    def __init__(self, output_dir: str = "./results/visualizations_3d"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Configure PyVista for headless/offscreen rendering
        pv.set_plot_theme("document")
        pv.OFF_SCREEN = True

    @staticmethod
    def _prepare_volume(vol: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(vol, torch.Tensor):
            arr = vol.detach().cpu().numpy()
        else:
            arr = np.copy(vol)

        if arr.ndim == 4: # (D, C, H, W)
            arr = arr[:, 0, :, :]

        # Normalize to [0, 1] for 3D visualization
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr.astype(np.float32)

    def render_sparse_vs_dense_3d(
        self,
        sparse_volume: Union[np.ndarray, torch.Tensor],
        dense_volume: Union[np.ndarray, torch.Tensor],
        gt_volume: Union[np.ndarray, torch.Tensor],
        volume_id: int,
        stride: int,
        save_name: Optional[str] = None,
    ) -> str:
        """
        Renders a 3-panel 3D figure comparing:
        1. Sparse input volume with visible slice gaps
        2. Diffusion reconstructed 3D volume
        3. Ground Truth dense 3D volume
        """
        sparse_arr = self._prepare_volume(sparse_volume)
        dense_arr = self._prepare_volume(dense_volume)
        gt_arr = self._prepare_volume(gt_volume)

        plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[1800, 600])

        # Panel 1: Sparse Volume
        plotter.subplot(0, 0)
        grid_sparse = pv.ImageData()
        grid_sparse.dimensions = sparse_arr.shape
        grid_sparse.point_data["values"] = sparse_arr.flatten(order="F")
        plotter.add_text(f"Sparse Input ({stride}x Stride)", font_size=12)
        plotter.add_volume(grid_sparse, cmap="bone", opacity="linear", shade=True)
        plotter.show_bounds(grid="front", location="outer", all_edges=True)

        # Panel 2: Diffusion Reconstructed Volume
        plotter.subplot(0, 1)
        grid_dense = pv.ImageData()
        grid_dense.dimensions = dense_arr.shape
        grid_dense.point_data["values"] = dense_arr.flatten(order="F")
        plotter.add_text("Diffusion Reconstructed 3D", font_size=12)
        plotter.add_volume(grid_dense, cmap="bone", opacity="linear", shade=True)
        plotter.show_bounds(grid="front", location="outer", all_edges=True)

        # Panel 3: Ground Truth Volume
        plotter.subplot(0, 2)
        grid_gt = pv.ImageData()
        grid_gt.dimensions = gt_arr.shape
        grid_gt.point_data["values"] = gt_arr.flatten(order="F")
        plotter.add_text("Ground Truth Dense 3D", font_size=12)
        plotter.add_volume(grid_gt, cmap="bone", opacity="linear", shade=True)
        plotter.show_bounds(grid="front", location="outer", all_edges=True)

        plotter.camera_position = "iso"

        if save_name is None:
            save_name = f"vol_3d_render_{volume_id}_stride_{stride}x.png"
        save_path = os.path.join(self.output_dir, save_name)
        plotter.screenshot(save_path)
        plotter.close()
        return save_path

    def render_ortho_slices_3d(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        title: str = "3D Orthogonal Slices",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Renders intersecting 3D orthogonal planes (Axial, Coronal, Sagittal) in 3D space.
        """
        arr = self._prepare_volume(volume)
        grid = pv.ImageData()
        grid.dimensions = arr.shape
        grid.point_data["values"] = arr.flatten(order="F")

        plotter = pv.Plotter(off_screen=True, window_size=[1000, 800])
        plotter.add_text(title, font_size=14)

        # Create orthogonal slices
        D, H, W = arr.shape
        slices = grid.slice_orthogonal(x=D // 2, y=H // 2, z=W // 2)
        plotter.add_mesh(slices, cmap="gray", show_scalar_bar=True)
        plotter.show_bounds(grid="front", location="outer", all_edges=True)
        plotter.camera_position = "iso"

        if save_name is None:
            save_name = "ortho_3d_slices.png"
        save_path = os.path.join(self.output_dir, save_name)
        plotter.screenshot(save_path)
        plotter.close()
        return save_path
