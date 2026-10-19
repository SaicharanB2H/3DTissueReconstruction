# Review findings

Reviewed 8 October 2026. Original user edits, presentations, disclosure template, trained checkpoints and historical benchmark files are retained.

## Correctness and reliability

| Priority | Finding | Resolution |
|---|---|---|
| High | Target-derived training statistics and full-ground-truth inference normalization expose missing intensity information | Shared statistics calculated only from acquired slices, separately for each volume and stride |
| High | Acquired targets participate in diffusion training and can appear in their own context | Diffusion slice dataset indexes missing targets only; VAE retains all targets |
| High | Reconstruction synthesizes indices implied by stride instead of actual acquired list | Missing indices are the complement of validated supplied indices; irregular acquisitions supported with boundary slices |
| High | Missing checkpoints silently permit random or frozen untrained models | Shared factory requires trained VAE and named diffusion states with strict state loading |
| Medium | EMA saved during training is not loaded for inference | Explicit raw/EMA selection; fine-tuning stores EMA for U-Net and context encoder |
| Medium | Resume compares against most recent loss rather than historical best; bounded runs lack explicit initialization controls | Best metric persisted/restored in checkpoint manager; new training CLI separates initialization from optimizer/RNG resume |
| Medium | Hardcoded 30x30 latent/context maps | Maps derive from encoder feature size; spatial input dimensions must be divisible by 8 |
| Medium | Explicit attention matrices waste GPU memory | PyTorch scaled-dot-product attention replaces equivalent attention computation, preserving parameter shapes |
| Medium | Multi-modal NIfTI transpose fails; identity affine implies undocumented geometry | Tested H,W,D,C export, normalized intensity and geometry sidecars, unknown qform/sform when geometry absent |
| Medium | Broad data ignore pattern hides source data package | Ignore only root data directory; src/data is visible to Git |
| Medium | 4096-slice cache can consume substantial memory per worker | Bound image cache to 128 entries; separate small statistics cache |
| Medium | Unknown schedules and invalid settings accepted; modality errors inconsistent | Validate schedules, sampling steps, dimensions, channels, acquisition indices and normalization bounds |
| Low | TensorBoard/tabulate dependencies omitted; optional logger fails without TensorBoard | Requirements corrected and optional logger guarded |

## Scientific and documentation corrections

- The historical `Bspline_3d` baseline is linear interpolation plus Gaussian smoothing (sigma 0.7 along depth), not a B-spline. New name: `smoothed_linear`; old names remain compatibility aliases.
- Historical benchmark covers ten held-out volumes and four strides, not all 56 held-out volumes. New comparisons use the same ten volumes at strides 8/16 with acquired-only preprocessing.
- Historical charts show incorrect acquired/missing counts and the original architecture image labels a 3D VAE/diffusion and CNN-or-ViT encoder. Actual models are 2D with a fixed residual convolutional encoder. Boundary-inclusive acquired/missing counts for D=155 are 78/77 at stride 2, 40/115 at stride 4, 21/134 at stride 8, and 11/144 at stride 16. New diagrams and disclosure use the implemented counts and architecture.
- PSNR and SSIM include image background and are averaged over 2D slices. They do not establish diagnostic accuracy, topology, or lesion preservation. Missing-slice metrics avoid inflation from exactly preserved acquired slices.
- Inspection of volume 100/slice 75 shows three mutually exclusive mask channels, contradicting the README's assumption that they are overlapping WT/TC/ET regions. Exact HDF5 class-to-label mapping requires provenance confirmation. Existing tumor visualization categories remain unverified, and ground-truth overlays do not validate synthesized tumor geometry.
- Demo visualization uses interpolation baselines, including linear reconstruction in 3D; its previous diffusion label was misleading and is corrected.
- The presentation guide describes diffusion panels that are absent from the saved 2D grids. Its slice-100 example is an acquired plane (all methods have PSNR 100), so it cannot demonstrate missing-slice synthesis. The old montage also has visibly poor intensity/axis presentation. These historical images are retained, with a new export and preview generated from the corrected selected evaluation.
- MRI scan acceleration, A100 deployment, absence of hallucinations, guaranteed smooth geometry, surgical accuracy, and measured cost savings are unsupported by saved validation. They are not carried into the disclosure as demonstrated benefits.
- VAE compression is 8x per spatial dimension and 16x in scalar element count for one-channel inputs. Cross-attention is present throughout U-Net blocks; VAE attention is at the bottleneck. ResNet18, AdaGN, perceptual losses, DDPM inference and several YAML switches were advertised without corresponding implementations.
- Local HDF5 images have no affine/spacing attributes. Official BraTS preprocessing is not sufficient to recover patient-specific orientation from these files.
- Existing best checkpoints: VAE epoch 13 (loss 0.0891816), diffusion epoch 48 (noise loss 0.1177198). These are different objectives and not reconstruction scores.

## Scope and remaining limits

Fine-tuning retains the architecture and pretrained VAE. Four-modality export is tested, but existing learned checkpoints are FLAIR-only. CT and other organs require separate datasets and training. HDF5 preprocessing/channel provenance, inventor identity, ownership/funding and public disclosure dates require confirmation. No new segmentation model, uncertainty calibration or clinical study is included.

Original slides and Blender assets are retained as historical materials; their unverified labels and claims are not evidence in the completed disclosure. The new README and disclosure are the corrected narrative.
