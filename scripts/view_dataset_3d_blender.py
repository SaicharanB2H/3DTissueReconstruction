"""
Direct BraTS Dataset 3D Brain & Tumor Object Generator for Blender.
Extracts complete 3D anatomical brain meshes and ground-truth 3D tumor sub-regions
directly from the raw BraTS2020 dataset for interactive 3D viewing and Blender modeling.
"""
import os
import sys
import argparse
import numpy as np
import pyvista as pv
import vtk
from scipy.ndimage import gaussian_filter

# Silence VTK low-level OpenGL framebuffer warnings on Windows/Optimus GPUs
vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.multi_samples = 4

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.dataset import BraTSVolumeDataset


def extract_surface_mesh(
    vol: np.ndarray,
    isovalue: float = 0.12,
    smooth_sigma: float = 0.8,
    smooth_iter: int = 40,
) -> pv.PolyData:
    """Extracts a smoothed 3D surface mesh from a 3D scalar volume using Marching Cubes."""
    if smooth_sigma > 0:
        vol_smooth = gaussian_filter(vol.astype(np.float32), sigma=smooth_sigma)
    else:
        vol_smooth = vol.astype(np.float32)

    grid = pv.ImageData()
    grid.dimensions = vol_smooth.shape
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data["values"] = vol_smooth.flatten(order="F")

    mesh = grid.contour([isovalue])
    if smooth_iter > 0 and mesh.n_points > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.05)
    return grid, mesh


def save_blender_scene(
    obj_path: str,
    objects: list,
):
    """
    Saves multiple meshes into a single Wavefront .obj file with a companion .mtl file.
    Creates separate 'o <name>' objects and material assignments ('usemtl') so Blender
    automatically imports them as distinct scene objects with materials pre-assigned!
    objects: list of tuples (name, PolyData mesh, material_name)
    """
    mtl_filename = os.path.basename(obj_path).replace(".obj", ".mtl")
    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_filename)

    # 1. Write companion .mtl material library
    with open(mtl_path, "w") as mf:
        mf.write("# Material Library for 3D Brain & Tumor\n\n")

        # Brain material: Translucent cortical flesh tone
        mf.write("newmtl BrainMaterial\n")
        mf.write("Ka 0.2 0.2 0.2\n")
        mf.write("Kd 0.85 0.76 0.70\n")
        mf.write("Ks 0.4 0.4 0.4\n")
        mf.write("Ns 30.0\n")
        mf.write("d 0.35\n")  # 35% opacity (translucent)
        mf.write("illum 2\n\n")

        # Whole tumor material: Bright anatomical crimson
        mf.write("newmtl TumorMaterial\n")
        mf.write("Ka 0.3 0.0 0.0\n")
        mf.write("Kd 0.95 0.12 0.12\n")
        mf.write("Ks 0.6 0.6 0.6\n")
        mf.write("Ns 45.0\n")
        mf.write("d 1.0\n")
        mf.write("illum 2\n\n")

        # Enhancing core material: Vibrant golden amber
        mf.write("newmtl CoreMaterial\n")
        mf.write("Ka 0.3 0.2 0.0\n")
        mf.write("Kd 0.98 0.82 0.10\n")
        mf.write("Ks 0.7 0.7 0.7\n")
        mf.write("Ns 50.0\n")
        mf.write("d 1.0\n")
        mf.write("illum 2\n\n")

    # 2. Write multi-object .obj file
    with open(obj_path, "w") as f:
        f.write("# 3D Brain & Tumor Multi-Object Mesh for Blender\n")
        f.write(f"mtllib {mtl_filename}\n\n")

        v_offset = 1
        for name, mesh, mat in objects:
            if mesh is None or mesh.n_points == 0:
                continue

            f.write(f"o {name}\n")
            f.write(f"g {name}\n")
            f.write(f"usemtl {mat}\n")

            pts = mesh.points
            for p in pts:
                f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

            # Extract triangular faces from PyVista PolyData
            faces = mesh.faces
            i = 0
            while i < len(faces):
                n_verts = faces[i]
                i += 1
                face_idx = faces[i : i + n_verts] + v_offset
                f.write(f"f {' '.join(str(idx) for idx in face_idx)}\n")
                i += n_verts

            v_offset += mesh.n_points
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D Brain & Tumor Blender Objects from BraTS Dataset")
    parser.add_argument(
        "--vol_id",
        type=int,
        default=100,
        help="Volume ID to extract from the dataset (default: 100)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./BraTS2020_training_data/content/data",
        help="Path to BraTS slice dataset folder",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="flair",
        choices=["flair", "t1", "t1ce", "t2"],
        help="MRI modality for brain tissue surface (default: flair)",
    )
    parser.add_argument(
        "--brain_threshold",
        type=float,
        default=0.12,
        help="Isovalue threshold for brain cortex surface (default: 0.12)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/3d_blender",
        help="Directory to save exported 3D Blender objects (.obj / .stl)",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Export 3D files without launching GUI window",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ppt_3d_dir = "./PPT/3d_models"
    os.makedirs(ppt_3d_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" BraTS Dataset 3D Blender Object Generator - Patient Volume {args.vol_id}")
    print("=" * 70)

    # 1. Load ground truth volume and multi-class tumor masks
    print(f"\n[1/4] Loading Patient Volume {args.vol_id} ({args.modality.upper()}) directly from dataset...")
    try:
        ds = BraTSVolumeDataset(args.data_dir, [args.vol_id], modality=args.modality)
        sample = ds[0]
    except Exception as e:
        print(f"Error loading volume {args.vol_id}: {e}")
        sys.exit(1)

    # Brain volume array: (155, 240, 240)
    vol_raw = sample["volume_raw"].numpy()[:, 0, :, :]

    # Robust percentile normalization to handle high-intensity outliers (e.g. contrast artifacts)
    pos_voxels = vol_raw[vol_raw > 0]
    if len(pos_voxels) > 0:
        p99 = float(np.percentile(pos_voxels, 99.0))
        vol_clipped = np.clip(vol_raw, 0, p99)
        vol_norm = vol_clipped / (p99 + 1e-8)
    else:
        vol_norm = (vol_raw - vol_raw.min()) / (vol_raw.max() - vol_raw.min() + 1e-8)

    # Tumor masks: (155, 3, 240, 240)
    # Channel 0: Whole Tumor (WT), Channel 1: Tumor Core (TC), Channel 2: Enhancing Tumor (ET)
    mask_arr = sample["mask_volume"].numpy()
    mask_wt = mask_arr[:, 0, :, :]  # Whole tumor
    mask_tc = mask_arr[:, 1, :, :]  # Tumor core
    mask_et = mask_arr[:, 2, :, :]  # Enhancing tumor

    print(f"      Volume Dimensions:     {vol_norm.shape} (155 slices x 240 x 240)")
    print(f"      Whole Tumor voxels:    {np.count_nonzero(mask_wt):,}")
    print(f"      Tumor Core voxels:     {np.count_nonzero(mask_tc):,}")
    print(f"      Enhancing Core voxels: {np.count_nonzero(mask_et):,}")

    # 2. Extract 3D Surface Meshes
    print(f"\n[2/4] Extracting 3D Polygonal Surface Meshes (Marching Cubes)...")
    print(f"      -> Extracting Brain Parenchyma Cortex (isovalue={args.brain_threshold})...")
    grid_brain, mesh_brain = extract_surface_mesh(
        vol_norm,
        isovalue=args.brain_threshold,
        smooth_sigma=0.8,
        smooth_iter=45,
    )
    print(f"         Brain Mesh: {mesh_brain.n_points:,} vertices, {mesh_brain.n_cells:,} faces")

    # Extract Whole Tumor 3D mesh if present
    mesh_tumor_wt = None
    if np.count_nonzero(mask_wt) > 10:
        print(f"      -> Extracting 3D Whole Tumor Mesh...")
        _, mesh_tumor_wt = extract_surface_mesh(
            mask_wt.astype(np.float32),
            isovalue=0.5,
            smooth_sigma=0.6,
            smooth_iter=30,
        )
        print(f"         Tumor Mesh: {mesh_tumor_wt.n_points:,} vertices, {mesh_tumor_wt.n_cells:,} faces")

    # Extract Enhancing Tumor 3D mesh if present
    mesh_tumor_et = None
    if np.count_nonzero(mask_et) > 10:
        print(f"      -> Extracting 3D Enhancing Tumor Core...")
        _, mesh_tumor_et = extract_surface_mesh(
            mask_et.astype(np.float32),
            isovalue=0.5,
            smooth_sigma=0.5,
            smooth_iter=20,
        )
        print(f"         Enhancing Core: {mesh_tumor_et.n_points:,} vertices, {mesh_tumor_et.n_cells:,} faces")

    # 3. Export 3D Blender Files (.OBJ & .STL)
    print(f"\n[3/4] Exporting 3D Model Files for Blender...")

    # Individual STL files
    brain_stl = os.path.join(args.output_dir, f"vol_{args.vol_id}_brain_surface.stl")
    mesh_brain.save(brain_stl)
    print(f"      -> Brain Surface (STL): {brain_stl}")

    if mesh_tumor_wt is not None:
        tumor_stl = os.path.join(args.output_dir, f"vol_{args.vol_id}_tumor_wt.stl")
        mesh_tumor_wt.save(tumor_stl)
        print(f"      -> Tumor Mesh (STL):    {tumor_stl}")

    # Combined Multi-Object Wavefront OBJ with .mtl materials (Best for Blender!)
    combined_obj = os.path.join(args.output_dir, f"vol_{args.vol_id}_dataset_brain_and_tumor.obj")
    mesh_list = [("Brain_Parenchyma", mesh_brain, "BrainMaterial")]
    if mesh_tumor_wt is not None:
        mesh_list.append(("Tumor_Whole", mesh_tumor_wt, "TumorMaterial"))
    if mesh_tumor_et is not None:
        mesh_list.append(("Tumor_Enhancing_Core", mesh_tumor_et, "CoreMaterial"))

    save_blender_scene(combined_obj, mesh_list)
    print(f"      -> Combined Multi-Object (OBJ): {combined_obj}")

    # Also copy to PPT/3d_models folder for presentations
    ppt_obj = os.path.join(ppt_3d_dir, f"vol_{args.vol_id}_dataset_brain_and_tumor.obj")
    save_blender_scene(ppt_obj, mesh_list)
    print(f"      -> Copied to presentation folder: {ppt_obj}")

    print("\n" + "*" * 70)
    print(f"  BLENDER USAGE INSTRUCTIONS:")
    print(f"  1. Open Blender.")
    print(f"  2. Go to: File -> Import -> Wavefront (.obj)")
    print(f"  3. Select: '{os.path.abspath(combined_obj)}'")
    print(f"  4. In Blender's Outliner (top-right), you will see:")
    print(f"     - 'Brain_Parenchyma' (with BrainMaterial pre-assigned)")
    print(f"     - 'Tumor_Whole' (with TumorMaterial pre-assigned)")
    print(f"     - 'Tumor_Enhancing_Core' (with CoreMaterial pre-assigned)")
    print(f"  5. Switch viewport shading to 'Material Preview' or 'Rendered'")
    print(f"     to see the translucent brain cortex with the tumor inside!")
    print("*" * 70)

    if args.export_only:
        print("\nExport complete. Exiting (--export_only was specified).")
        return

    # 4. Interactive 3D Window (Blender Style)
    print(f"\n[4/4] Launching Interactive 3D Viewer...")
    print("\n" + "=" * 65)
    print(" Controls (Blender Style):")
    print(" - Left Click + Drag : 360° Orbit / Rotate around the 3D Brain")
    print(" - Scroll Wheel      : Zoom in / out")
    print(" - Shift + Left Click: Pan camera")
    print(" - Press 'r'         : Reset camera")
    print(" - Press 'q'         : Close 3D window")
    print("=" * 65 + "\n")

    pv.set_plot_theme("dark")
    plotter = pv.Plotter(window_size=[1300, 850], title=f"BraTS Volume {args.vol_id}: Complete 3D Brain & Tumor")

    # Add semi-transparent brain cortex (allows looking inside at the tumor!)
    plotter.add_mesh(
        mesh_brain,
        color="#D4B8A5",
        opacity=0.35,
        smooth_shading=True,
        specular=0.5,
        label="Brain Cortex (Translucent)",
    )

    # Add solid glowing red tumor nested inside the brain
    if mesh_tumor_wt is not None:
        plotter.add_mesh(
            mesh_tumor_wt,
            color="#FF2A2A",
            opacity=1.0,
            smooth_shading=True,
            specular=0.8,
            specular_power=20,
            label="Tumor (Whole Tumor)",
        )

    # Add bright yellow core if present
    if mesh_tumor_et is not None:
        plotter.add_mesh(
            mesh_tumor_et,
            color="#FFD700",
            opacity=1.0,
            smooth_shading=True,
            specular=0.9,
            label="Enhancing Tumor Core",
        )

    plotter.add_legend(bcolor="#1C1C1E", border=True, size=(0.28, 0.15))
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = "iso"
    plotter.show()


if __name__ == "__main__":
    main()
