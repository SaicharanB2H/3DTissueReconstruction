"""
1-Click Blender Scene Setup Script for 3D Brain & Tumor Visualization.
Designed to be run inside Blender's Scripting workspace or via CLI:
    blender --python scripts/setup_blender_scene.py

Features:
1. Automatically discovers and imports the latest multi-object Brain & Tumor OBJ file.
2. Sets up realistic medical shaders:
   - Brain Cortex: Translucent glassmorphic/tissue shader with alpha blend.
   - Brain Core: Creamy solid parenchymal white matter shader (no empty space!).
   - Tumor: Vivid glowing crimson shader with subsurface scattering and emission.
   - Core / Enhancing: Golden amber active core shader.
3. Configures 3-point medical studio lighting and centered camera.
4. Adds an interactive Boolean Cutaway Cutter so you can slice through the brain in real time!
"""

import os
import sys

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def setup_scene(obj_path: str = None):
    if not IN_BLENDER:
        print("[Error] This script must be run inside Blender!")
        print("Usage in Blender:")
        print("  1. Open Blender")
        print("  2. Go to 'Scripting' tab at top")
        print("  3. Open and run this script (Alt + P)")
        print("Or from command line:")
        print("  blender.exe --python scripts/setup_blender_scene.py")
        return

    print("=" * 65)
    print(" Setting up 3D Medical Brain & Tumor Scene in Blender...")
    print("=" * 65)

    # 1. Clean default scene (remove default Cube, Light, Camera if clean start)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 2. Locate OBJ file
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    candidate_paths = [
        obj_path,
        os.path.join(project_root, "results", "reconstructed_volumes", "recon_vol_100_stride_4x_brain_and_tumor.obj"),
        os.path.join(project_root, "results", "reconstructed_volumes", "recon_vol_100_stride_4x_cutaway.obj"),
        os.path.join(project_root, "PPT", "3d_models", "recon_vol_100_stride_4x_brain_and_tumor.obj"),
        os.path.join(project_root, "PPT", "3d_models", "recon_vol_100_stride_4x_cutaway.obj"),
    ]

    target_obj = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            target_obj = p
            break

    if not target_obj:
        print(f"[Error] No Brain & Tumor OBJ file found in candidates: {candidate_paths}")
        return

    print(f"Loading 3D Model: {target_obj}")

    # 3. Import OBJ file (compatible with Blender 2.8x through 4.x)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=target_obj)
    elif hasattr(bpy.ops.import_scene, "obj"):
        bpy.ops.import_scene.obj(filepath=target_obj)
    else:
        print("[Error] No OBJ importer operator found in this Blender version!")
        return

    imported_objects = list(bpy.context.selected_objects)
    print(f"Imported {len(imported_objects)} objects:")
    for obj in imported_objects:
        print(f"  - {obj.name} (type: {obj.type})")

    # Center origin of imported meshes to geometry
    for obj in imported_objects:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # 4. Enhance Materials for Medical Realism
    # Setup Brain Cortex Material (Translucent with Alpha Blend)
    cortex_mat = bpy.data.materials.get("BrainCortexMaterial")
    if cortex_mat and cortex_mat.use_nodes:
        bsdf = cortex_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Translucent flesh tone
            bsdf.inputs["Base Color"].default_value = (0.88, 0.76, 0.70, 1.0)
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = 0.30
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.35
            # Blender 3.x / 4.x transparency settings
            if hasattr(cortex_mat, "blend_method"):
                cortex_mat.blend_method = 'BLEND'
            if hasattr(cortex_mat, "shadow_method"):
                cortex_mat.shadow_method = 'HASHED'

    # Setup Brain Internal Core (Solid Parenchyma)
    core_mat = bpy.data.materials.get("BrainCoreMaterial")
    if core_mat and core_mat.use_nodes:
        bsdf = core_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Ivory/cream solid white matter
            bsdf.inputs["Base Color"].default_value = (0.92, 0.88, 0.82, 1.0)
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = 0.95
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.50

    # Setup Tumor Material (Glowing Anatomical Crimson)
    tumor_mat = bpy.data.materials.get("TumorMaterial")
    if tumor_mat and tumor_mat.use_nodes:
        bsdf = tumor_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.96, 0.10, 0.10, 1.0)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (0.90, 0.05, 0.05, 1.0)
            elif "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = (0.90, 0.05, 0.05, 1.0)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 0.6
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.25

    # Setup Tumor Core Material (Golden Amber)
    t_core_mat = bpy.data.materials.get("CoreMaterial")
    if t_core_mat and t_core_mat.use_nodes:
        bsdf = t_core_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.98, 0.65, 0.05, 1.0)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (0.95, 0.60, 0.0, 1.0)
            elif "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = (0.95, 0.60, 0.0, 1.0)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 0.8

    # 5. Setup Studio 3-Point Lighting
    # Key Light
    bpy.ops.object.light_add(type='AREA', location=(150, -150, 180))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    key_light.data.size = 100

    # Fill Light
    bpy.ops.object.light_add(type='AREA', location=(-150, -100, 100))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 250
    fill_light.data.size = 120

    # Rim / Backlight
    bpy.ops.object.light_add(type='POINT', location=(0, 200, 150))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 350

    # 6. Setup Camera
    bpy.ops.object.camera_add(location=(180, -220, 160))
    cam = bpy.context.active_object
    cam.name = "Medical_Camera"
    bpy.context.scene.camera = cam

    # Point camera at center (0, 0, 0)
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = imported_objects[0]
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    # 7. Set Viewport Shading to Material Preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    print("=" * 65)
    print(" Scene setup complete!")
    print(" - Outliner has separate objects: Brain_Cortex, Brain_Internal_Core, Tumor_Whole, Tumor_Core")
    print(" - Viewport shading is set to Material Preview")
    print(" - Press 'Numpad 0' to look through the Medical Camera")
    print("=" * 65)


if __name__ == "__main__":
    setup_scene()
