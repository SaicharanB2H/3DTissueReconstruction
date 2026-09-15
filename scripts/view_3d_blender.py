"""
Interactive 3D Brain Visualizer & 3D Mesh Exporter (Blender/STL/OBJ).
Features:
1. Marching Cubes 3D Surface Extraction: converts the reconstructed MRI volume
   into a real 3D polygonal mesh with realistic lighting & shading.
2. Direct 3D Mesh Export (.obj & .stl): open directly in Blender, MeshLab, or Windows 3D Viewer!
3. Interactive 3D PyVista Viewer: orbit (360° rotate), zoom, and pan like in Blender.
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pyvista as pv
from scipy.ndimage import gaussian_filter


def load_volume(path: str) -> np.ndarray:
    """Loads NIfTI volume, ensures (D, H, W) orientation, and normalizes to [0, 1]."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[0] if data.shape[0] == 1 else data[..., 0]
    # If (H, W, D), transpose to (D, H, W)
    if data.shape[-1] < min(data.shape[0], data.shape[1]):
        data = np.transpose(data, (2, 0, 1))

    # Normalize to [0, 1]
    vmin, vmax = data.min(), data.max()
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin)
    return data


def create_3d_mesh(vol: np.ndarray, isovalue: float = 0.22, smooth_sigma: float = 0.8) -> pv.PolyData:
    """
    Extracts a 3D surface mesh using Marching Cubes.
    Smooths slightly with Gaussian filter to reduce voxel stepping artifacts.
    """
    # Smooth slightly for smooth organic brain surface
    smoothed = gaussian_filter(vol, sigma=smooth_sigma)

    # Wrap into PyVista ImageData
    grid = pv.ImageData()
    grid.dimensions = smoothed.shape
    # Spacing in mm (BraTS typical spacing 1x1x1 mm)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data["values"] = smoothed.flatten(order="F")

    # Marching cubes contour extraction
    mesh = grid.contour([isovalue])
    # Smooth mesh surface
    mesh = mesh.smooth(n_iter=50, relaxation_factor=0.05)
    return grid, mesh


def main():
    parser = argparse.ArgumentParser(description="3D Brain Mesh & Interactive Viewer for Blender/PyVista")
    parser.add_argument(
        "--file",
        type=str,
        default="./results/reconstructed_volumes/recon_vol_100_stride_4x.nii.gz",
        help="Path to .nii.gz volume",
    )
    parser.add_argument(
        "--isovalue",
        type=float,
        default=0.22,
        help="Surface threshold for brain tissue extraction (default: 0.22)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mesh", "volume", "both"],
        default="mesh",
        help="Visualization mode: 'mesh' (Blender-like surface), 'volume' (raycasted), or 'both'",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Only export .obj and .stl without opening GUI window",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    print(f"\n[1/3] Loading reconstructed volume: {args.file}")
    vol = load_volume(args.file)
    print(f"      Volume dimensions: {vol.shape} (D={vol.shape[0]}, H={vol.shape[1]}, W={vol.shape[2]})")

    print(f"[2/3] Extracting 3D Marching Cubes surface mesh (isovalue={args.isovalue})...")
    grid, mesh = create_3d_mesh(vol, isovalue=args.isovalue)
    print(f"      Extracted 3D Mesh: {mesh.n_points:,} vertices, {mesh.n_cells:,} faces")

    # Export to .obj and .stl files
    base_no_ext = args.file.replace(".nii.gz", "").replace(".nii", "")
    obj_path = f"{base_no_ext}_mesh.obj"
    stl_path = f"{base_no_ext}_mesh.stl"

    print(f"[3/3] Exporting 3D Mesh for Blender:")
    mesh.save(stl_path)
    print(f"      -> STL format: {stl_path}")
    mesh.save(obj_path)
    print(f"      -> OBJ format: {obj_path}")
    print(f"\n  ==> You can now drag '{os.path.basename(obj_path)}' directly into Blender or Windows 3D Viewer!")

    if args.export_only:
        print("\nExport completed. Exiting (--export_only was set).")
        return

    # Launch Interactive 3D Window
    print("\n" + "=" * 65)
    print(" Interactive 3D Controls (Blender style):")
    print(" - Left Click + Drag : 360° Rotate / Orbit around the 3D Brain")
    print(" - Scroll Wheel      : Zoom in / out")
    print(" - Shift + Left Click: Pan the view")
    print(" - Press 'r'         : Reset camera")
    print(" - Press 'q'         : Close 3D window")
    print("=" * 65 + "\n")

    # Interactive window setup
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(window_size=[1280, 800], title="3D Reconstructed Brain Viewer")

    if args.mode in ["mesh", "both"]:
        # Anatomical brain surface shading
        plotter.add_mesh(
            mesh,
            color="#E8C9B8",  # Realistic brain cortex flesh tone
            specular=0.4,
            specular_power=15,
            smooth_shading=True,
            opacity=0.92 if args.mode == "both" else 1.0,
            show_edges=False,
            label="Reconstructed Brain Surface",
        )

    if args.mode in ["volume", "both"]:
        # Semi-transparent volumetric raycasting
        opacity_transfer = [0.0, 0.0, 0.1, 0.4, 0.8, 1.0]
        plotter.add_volume(
            grid,
            cmap="bone",
            opacity=opacity_transfer,
            shade=True,
        )

    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = "iso"
    plotter.enable_anti_aliasing()
    plotter.show()


if __name__ == "__main__":
    main()
