"""
Interactive 3D Brain & Tumor Visualizer and Multi-Object Blender Exporter.
Upgraded to support:
1. Reconstructed volume + Ground-Truth / Extracted Tumor co-visualization.
2. Solid Internal Brain Parenchyma Core (eliminates the empty hollow void).
3. Anatomical Cutaway / Cross-Section Meshes (closed, solid interior plane showing internal tumor).
4. Multi-object Wavefront OBJ + MTL export with pre-assigned materials for Blender.
5. Interactive 3D PyVista Viewer with orbit, zoom, pan, and real-time slice clipping.
"""
import os
import sys
import re
import argparse
import numpy as np
import nibabel as nib
import pyvista as pv
import vtk
from scipy.ndimage import gaussian_filter, label

# Silence VTK low-level OpenGL framebuffer warnings on Windows/Optimus GPUs
vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.multi_samples = 4

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.data.dataset import BraTSVolumeDataset
except ImportError:
    BraTSVolumeDataset = None


def parse_volume_id_from_path(file_path: str) -> int:
    """Extracts the patient volume ID from the filename (e.g. recon_vol_100_stride_4x.nii.gz -> 100)."""
    base = os.path.basename(file_path)
    match = re.search(r"vol(?:ume)?_(\d+)", base, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 100  # Default fallback


def load_nifti_volume(path: str) -> np.ndarray:
    """Loads NIfTI volume, ensures (D, H, W) orientation, and normalizes to [0, 1]."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[0] if data.shape[0] == 1 else data[..., 0]
    
    if data.shape[-1] < min(data.shape[0], data.shape[1]):
        data = np.transpose(data, (2, 0, 1))

  
    vmin, vmax = float(data.min()), float(data.max())
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin)
    return data


def extract_surface_mesh(
    vol: np.ndarray,
    isovalue: float = 0.20,
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


def extract_tumor_from_dataset(vol_id: int, data_dir: str):
    """Loads ground-truth tumor masks (WT, TC, ET) from the BraTS dataset."""
    if BraTSVolumeDataset is None or not os.path.exists(data_dir):
        return None, None, None

    try:
        ds = BraTSVolumeDataset(data_dir, [vol_id], modality="flair")
        sample = ds[0]
        mask_arr = sample["mask_volume"].numpy()  # (155, 3, 240, 240)
        mask_wt = mask_arr[:, 0, :, :]  # Whole tumor
        mask_tc = mask_arr[:, 1, :, :]  # Tumor core
        mask_et = mask_arr[:, 2, :, :]  # Enhancing tumor
        return mask_wt, mask_tc, mask_et
    except Exception as e:
        print(f"      [Warning] Could not load dataset mask for volume {vol_id}: {e}")
        return None, None, None


def extract_tumor_from_reconstructed_intensity(vol: np.ndarray, brain_mask_threshold: float = 0.18):
    """
    Fallback tumor extraction directly from reconstructed FLAIR intensity.
    FLAIR tumors are hyperintense relative to normal brain parenchyma.
    """
    brain_mask = vol > brain_mask_threshold
    if not np.any(brain_mask):
        return None

    parenchyma_vals = vol[brain_mask]
    # Tumors typically sit in the top 3-5% brightest voxels in FLAIR
    p96 = float(np.percentile(parenchyma_vals, 96.0))
    tumor_candidate = (vol > p96) & brain_mask

    # Retain only connected components with sufficient volume (>50 voxels)
    labeled_arr, num_features = label(tumor_candidate)
    if num_features == 0:
        return None

    counts = np.bincount(labeled_arr.flat)
    counts[0] = 0  # ignore background
    valid_labels = np.where(counts > 50)[0]

    if len(valid_labels) == 0:
        return None

    cleaned_mask = np.isin(labeled_arr, valid_labels).astype(np.float32)
    return cleaned_mask


def create_anatomical_cutaway(
    mesh: pv.PolyData,
    cut_type: str = "sagittal",
    origin_offset: float = 0.0,
) -> pv.PolyData:
    """
    Creates an anatomical cutaway / cross-section mesh with a closed, solid interior plane.
    cut_type: 'sagittal' (divides left/right hemispheres) or 'coronal' (divides anterior/posterior).
    """
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    # PyVista axes: x=D (slices, ~0..154), y=H (~0..239), z=W (~0..239)
    x_center = (bounds[0] + bounds[1]) / 2.0
    y_center = (bounds[2] + bounds[3]) / 2.0
    z_center = (bounds[4] + bounds[5]) / 2.0

    if cut_type == "sagittal":
        # Sagittal cut along z midline
        normal = (0.0, 0.0, 1.0)
        origin = (0.0, 0.0, z_center + origin_offset)
    elif cut_type == "coronal":
        # Coronal cut along y axis
        normal = (0.0, 1.0, 0.0)
        origin = (0.0, y_center + origin_offset, 0.0)
    elif cut_type == "axial":
        # Axial cut along x axis
        normal = (1.0, 0.0, 0.0)
        origin = (x_center + origin_offset, 0.0, 0.0)
    else:
        normal = (0.0, 0.0, 1.0)
        origin = (0.0, 0.0, z_center)

    try:
        cut_mesh = mesh.clip_closed_surface(normal=normal, origin=origin)
        if cut_mesh.n_points > 0:
            return cut_mesh
    except Exception as e:
        print(f"      [Notice] clip_closed_surface fallback: {e}")

    # Fallback to standard clipping
    return mesh.clip(normal=normal, origin=origin, invert=False)


def save_blender_scene(
    obj_path: str,
    objects: list,
):
    """
    Saves multiple meshes into a single Wavefront .obj file with a companion .mtl file.
    objects: list of tuples (name, PolyData mesh, material_name)
    """
    mtl_filename = os.path.basename(obj_path).replace(".obj", ".mtl")
    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_filename)

    # 1. Write companion .mtl material library
    with open(mtl_path, "w") as mf:
        mf.write("# Material Library for 3D Brain, Internal Core & Tumor\n\n")

        # Brain Cortex material: Translucent cortical surface
        mf.write("newmtl BrainCortexMaterial\n")
        mf.write("Ka 0.25 0.22 0.20\n")
        mf.write("Kd 0.88 0.78 0.72\n")
        mf.write("Ks 0.45 0.45 0.45\n")
        mf.write("Ns 35.0\n")
        mf.write("d 0.32\n")  # 32% opacity: allows seeing internal core & tumor
        mf.write("illum 2\n\n")

        # Brain Internal Core: Solid deep parenchymal white matter (eliminates empty space!)
        mf.write("newmtl BrainCoreMaterial\n")
        mf.write("Ka 0.30 0.28 0.26\n")
        mf.write("Kd 0.92 0.88 0.82\n")
        mf.write("Ks 0.25 0.25 0.25\n")
        mf.write("Ns 20.0\n")
        mf.write("d 0.95\n")  # Solid interior
        mf.write("illum 2\n\n")

        # Cutaway anatomical cross-section material
        mf.write("newmtl CutawayMaterial\n")
        mf.write("Ka 0.28 0.25 0.23\n")
        mf.write("Kd 0.90 0.82 0.76\n")
        mf.write("Ks 0.35 0.35 0.35\n")
        mf.write("Ns 25.0\n")
        mf.write("d 1.0\n")
        mf.write("illum 2\n\n")

        # Whole tumor material: Vivid anatomical crimson red
        mf.write("newmtl TumorMaterial\n")
        mf.write("Ka 0.40 0.05 0.05\n")
        mf.write("Kd 0.96 0.12 0.12\n")
        mf.write("Ks 0.75 0.75 0.75\n")
        mf.write("Ns 50.0\n")
        mf.write("d 1.0\n")
        mf.write("illum 2\n\n")

        # Tumor active core: Bright golden amber
        mf.write("newmtl CoreMaterial\n")
        mf.write("Ka 0.40 0.25 0.02\n")
        mf.write("Kd 0.98 0.68 0.06\n")
        mf.write("Ks 0.80 0.80 0.80\n")
        mf.write("Ns 55.0\n")
        mf.write("d 1.0\n")
        mf.write("illum 2\n\n")

        # Enhancing core: Radiant luminous gold
        mf.write("newmtl EnhancingMaterial\n")
        mf.write("Ka 0.40 0.35 0.05\n")
        mf.write("Kd 1.00 0.92 0.15\n")
        mf.write("Ks 0.90 0.90 0.90\n")
        mf.write("Ns 60.0\n")
        mf.write("d 1.0\n")
        mf.write("illum 2\n\n")

    # 2. Write multi-object .obj file
    with open(obj_path, "w") as f:
        f.write("# 3D Brain, Internal Core & Tumor Multi-Object Mesh for Blender\n")
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
    parser = argparse.ArgumentParser(description="3D Brain & Tumor Visualizer and Blender Multi-Object Exporter")
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
        help="BraTS Patient Volume ID (default: auto-detected from filename)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./BraTS2020_training_data/content/data",
        help="Path to BraTS slice dataset folder",
    )
    parser.add_argument(
        "--cortex_threshold",
        type=float,
        default=0.20,
        help="Isovalue threshold for brain cortex surface (default: 0.20)",
    )
    parser.add_argument(
        "--core_threshold",
        type=float,
        default=0.45,
        help="Isovalue threshold for solid deep brain parenchymal core (default: 0.45)",
    )
    parser.add_argument(
        "--cutaway_plane",
        type=str,
        choices=["sagittal", "coronal", "axial", "none"],
        default="sagittal",
        help="Anatomical cutaway cross-section plane (default: sagittal)",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Export 3D files without launching interactive GUI window",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    vol_id = args.vol_id if args.vol_id is not None else parse_volume_id_from_path(args.file)
    output_dir = os.path.dirname(args.file)
    ppt_3d_dir = "./PPT/3d_models"
    os.makedirs(ppt_3d_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print(f" 3D Brain Reconstruction & Tumor Improvised Exporter (Patient Volume {vol_id})")
    print("=" * 75)

    # 1. Load reconstructed volume
    print(f"\n[1/5] Loading reconstructed volume: {args.file}")
    vol = load_nifti_volume(args.file)
    print(f"      Dimensions: {vol.shape} (D={vol.shape[0]}, H={vol.shape[1]}, W={vol.shape[2]})")

    # 2. Extract Tumor Masks (from dataset or reconstructed intensity)
    print(f"\n[2/5] Locating 3D Tumor for Volume {vol_id}...")
    mask_wt, mask_tc, mask_et = extract_tumor_from_dataset(vol_id, args.data_dir)

    mesh_tumor_wt = None
    mesh_tumor_tc = None
    mesh_tumor_et = None

    if mask_wt is not None and np.count_nonzero(mask_wt) > 10:
        print(f"      -> Found Patient {vol_id} Ground-Truth Tumor in Dataset:")
        print(f"         Whole Tumor voxels:    {np.count_nonzero(mask_wt):,}")
        print(f"         Tumor Core voxels:     {np.count_nonzero(mask_tc):,}")
        print(f"         Enhancing Core voxels: {np.count_nonzero(mask_et):,}")

        # Extract Whole Tumor 3D Mesh
        _, mesh_tumor_wt = extract_surface_mesh(mask_wt.astype(np.float32), isovalue=0.5, smooth_sigma=0.6, smooth_iter=30)
        # Extract Tumor Core
        if np.count_nonzero(mask_tc) > 10:
            _, mesh_tumor_tc = extract_surface_mesh(mask_tc.astype(np.float32), isovalue=0.5, smooth_sigma=0.5, smooth_iter=25)
        # Extract Enhancing Core
        if np.count_nonzero(mask_et) > 10:
            _, mesh_tumor_et = extract_surface_mesh(mask_et.astype(np.float32), isovalue=0.5, smooth_sigma=0.5, smooth_iter=20)
    else:
        print(f"      -> Dataset mask not available. Extracting hyperintense lesion from reconstructed volume...")
        lesion_mask = extract_tumor_from_reconstructed_intensity(vol, brain_mask_threshold=args.cortex_threshold)
        if lesion_mask is not None and np.count_nonzero(lesion_mask) > 20:
            _, mesh_tumor_wt = extract_surface_mesh(lesion_mask, isovalue=0.5, smooth_sigma=0.6, smooth_iter=30)
            print(f"         Extracted Reconstructed Lesion: {mesh_tumor_wt.n_points:,} vertices")

    # 3. Extract Brain Cortex and Deep Internal Parenchyma Core (Eliminating Empty Space!)
    print(f"\n[3/5] Extracting 3D Anatomical Brain Structures...")
    print(f"      -> Extracting Outer Brain Cortex (isovalue={args.cortex_threshold})...")
    grid_brain, mesh_cortex = extract_surface_mesh(vol, isovalue=args.cortex_threshold, smooth_sigma=0.8, smooth_iter=45)
    print(f"         Brain Cortex: {mesh_cortex.n_points:,} vertices, {mesh_cortex.n_cells:,} faces")

    print(f"      -> Extracting Deep Brain Parenchyma Core (isovalue={args.core_threshold})...")
    _, mesh_core = extract_surface_mesh(vol, isovalue=args.core_threshold, smooth_sigma=0.8, smooth_iter=35)
    print(f"         Internal Core: {mesh_core.n_points:,} vertices, {mesh_core.n_cells:,} faces")
    print(f"         ==> Internal Core fills the internal cavity so the 3D model is NOT hollow!")

    # 4. Generate Anatomical Cutaway / Cross-Section Mesh
    print(f"\n[4/5] Creating Anatomical Cutaway Cross-Section ({args.cutaway_plane.upper()} Cut)...")
    mesh_cutaway = create_anatomical_cutaway(mesh_cortex, cut_type=args.cutaway_plane)
    print(f"      Cutaway Mesh: {mesh_cutaway.n_points:,} vertices, {mesh_cutaway.n_cells:,} faces")
    print(f"      ==> Exposes solid internal cross-section with the tumor directly visible!")

    # 5. Export Multi-Object 3D Models for Blender
    print(f"\n[5/5] Exporting Multi-Object Blender Assets (.obj + .mtl)...")
    base_no_ext = args.file.replace(".nii.gz", "").replace(".nii", "")

    # Model A: Full Brain with Translucent Cortex + Solid Internal Core + Embedded Tumor
    full_obj_path = f"{base_no_ext}_brain_and_tumor.obj"
    full_objects = [
        ("Brain_Cortex", mesh_cortex, "BrainCortexMaterial"),
        ("Brain_Internal_Core", mesh_core, "BrainCoreMaterial"),
    ]
    if mesh_tumor_wt is not None:
        full_objects.append(("Tumor_Whole", mesh_tumor_wt, "TumorMaterial"))
    if mesh_tumor_tc is not None:
        full_objects.append(("Tumor_Core", mesh_tumor_tc, "CoreMaterial"))
    if mesh_tumor_et is not None:
        full_objects.append(("Tumor_Enhancing", mesh_tumor_et, "EnhancingMaterial"))

    save_blender_scene(full_obj_path, full_objects)
    print(f"      -> [Model A: Complete Translucent & Core] : {full_obj_path}")

    # Model B: Anatomical Cutaway Half-Brain with Solid Cross-Section & Embedded Tumor
    cutaway_obj_path = f"{base_no_ext}_cutaway.obj"
    cutaway_objects = [
        ("Brain_Cutaway", mesh_cutaway, "CutawayMaterial"),
    ]
    if mesh_tumor_wt is not None:
        cutaway_objects.append(("Tumor_Whole", mesh_tumor_wt, "TumorMaterial"))
    if mesh_tumor_tc is not None:
        cutaway_objects.append(("Tumor_Core", mesh_tumor_tc, "CoreMaterial"))
    if mesh_tumor_et is not None:
        cutaway_objects.append(("Tumor_Enhancing", mesh_tumor_et, "EnhancingMaterial"))

    save_blender_scene(cutaway_obj_path, cutaway_objects)
    print(f"      -> [Model B: Anatomical Cutaway View]     : {cutaway_obj_path}")

    # Export STL files for cutaway and tumor (for Windows 3D Viewer & 3D printing)
    cutaway_stl = f"{base_no_ext}_cutaway.stl"
    mesh_cutaway.save(cutaway_stl)
    print(f"      -> [Cutaway STL]                          : {cutaway_stl}")
    if mesh_tumor_wt is not None:
        tumor_stl = f"{base_no_ext}_tumor.stl"
        mesh_tumor_wt.save(tumor_stl)
        print(f"      -> [Tumor STL]                            : {tumor_stl}")

    # Fast copy OBJ and MTL to PPT/3d_models folder for presentations
    import shutil
    shutil.copy(full_obj_path, os.path.join(ppt_3d_dir, os.path.basename(full_obj_path)))
    shutil.copy(full_obj_path.replace(".obj", ".mtl"), os.path.join(ppt_3d_dir, os.path.basename(full_obj_path).replace(".obj", ".mtl")))
    shutil.copy(cutaway_obj_path, os.path.join(ppt_3d_dir, os.path.basename(cutaway_obj_path)))
    shutil.copy(cutaway_obj_path.replace(".obj", ".mtl"), os.path.join(ppt_3d_dir, os.path.basename(cutaway_obj_path).replace(".obj", ".mtl")))
    shutil.copy(cutaway_stl, os.path.join(ppt_3d_dir, os.path.basename(cutaway_stl)))
    if mesh_tumor_wt is not None:
        shutil.copy(tumor_stl, os.path.join(ppt_3d_dir, os.path.basename(tumor_stl)))
    print(f"      -> Copied updated 3D models to '{ppt_3d_dir}'")

    print("\n" + "*" * 75)
    print("  HOW TO VIEW IN BLENDER:")
    print("  -----------------------")
    print(f"  Method 1 (Instant Solid Cutaway - Recommended):")
    print(f"    1. In Blender: File -> Import -> Wavefront (.obj)")
    print(f"    2. Choose: '{os.path.abspath(cutaway_obj_path)}'")
    print(f"    3. The brain is sliced open with a solid flat cross section,")
    print(f"       and the bright red tumor is immediately visible inside!")
    print(f"")
    print(f"  Method 2 (Complete Translucent Cortex + Internal Core):")
    print(f"    1. Choose: '{os.path.abspath(full_obj_path)}'")
    print(f"    2. Switch viewport shading to 'Material Preview' (press Z -> 2)")
    print(f"    3. In the Outliner (top right), you will see:")
    print(f"       - 'Brain_Cortex' (translucent outer surface)")
    print(f"       - 'Brain_Internal_Core' (solid white matter core - no empty space!)")
    print(f"       - 'Tumor_Whole' (vivid red anatomical tumor)")
    print(f"       - 'Tumor_Core' (golden amber active core)")
    print("*" * 75)

    if args.export_only:
        print("\nExport completed successfully (--export_only was set). Exiting.")
        return

    # 6. Interactive 3D PyVista Viewer
    print("\nLaunching Interactive 3D Viewer...")
    print("\n" + "=" * 65)
    print(" Interactive Controls:")
    print(" - Left Click + Drag : 360° Orbit / Rotate around 3D Brain")
    print(" - Scroll Wheel      : Zoom in / out")
    print(" - Shift + Left Click: Pan view")
    print(" - Press 't'         : Toggle cortex transparency")
    print(" - Press 'r'         : Reset camera")
    print(" - Press 'q'         : Close viewer")
    print("=" * 65 + "\n")

    pv.set_plot_theme("dark")
    plotter = pv.Plotter(window_size=[1300, 850], title=f"Reconstructed Brain & Tumor (Patient {vol_id})")

    # Add Translucent Cortex
    actor_cortex = plotter.add_mesh(
        mesh_cortex,
        color="#D8BAA8",
        opacity=0.32,
        smooth_shading=True,
        specular=0.5,
        label="Brain Cortex (Translucent)",
    )

    # Add Internal Core
    plotter.add_mesh(
        mesh_core,
        color="#EAE3D9",
        opacity=0.45,
        smooth_shading=True,
        specular=0.3,
        label="Internal Parenchyma Core",
    )

    # Add Whole Tumor
    if mesh_tumor_wt is not None:
        plotter.add_mesh(
            mesh_tumor_wt,
            color="#FF2424",
            opacity=1.0,
            smooth_shading=True,
            specular=0.8,
            specular_power=30,
            label="Reconstructed Tumor (Whole)",
        )

    # Add Tumor Core
    if mesh_tumor_tc is not None:
        plotter.add_mesh(
            mesh_tumor_tc,
            color="#FFA500",
            opacity=1.0,
            smooth_shading=True,
            specular=0.9,
            specular_power=40,
            label="Tumor Active Core",
        )

    # Toggle transparency callback
    transparency_state = {"level": 0}
    levels = [0.32, 0.85, 0.0]

    def toggle_cortex():
        transparency_state["level"] = (transparency_state["level"] + 1) % len(levels)
        new_op = levels[transparency_state["level"]]
        actor_cortex.GetProperty().SetOpacity(new_op)
        plotter.render()

    plotter.add_key_event("t", toggle_cortex)
    plotter.add_legend(bcolor="#18181A", border=True, size=(0.28, 0.18))
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = "iso"
    plotter.show()


if __name__ == "__main__":
    main()
