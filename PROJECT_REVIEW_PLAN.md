# Approved project review and improvement plan

Approved 8 October 2026. Primary scope: BraTS FLAIR MRI; prioritize strides 8 and 16.

## Deliverables and sequence

1. Review source, tests, configurations, presentations, PDFs, dataset metadata, checkpoints and logs. Record findings with evidence.
2. Correct acquired-only normalization, missing-target training, acquired-index reconstruction, checkpoint loading, EMA/resume, configuration and exports. Preserve original artifacts and user edits.
3. Run regression tests, a training/resume smoke run and end-to-end reconstruction.
4. Run local GPU experiments in a separate output directory: 20 minutes reference evaluation, 70 minutes fine-tuning, 30 minutes selection and final comparisons (maximum two hours total).
5. Fine-tune existing encoder/U-Net at learning rate 2e-5, frozen VAE, equal sampling of strides 4/8/16. Compare raw/EMA weights, 25/50 DDIM steps and smoothing on/off on five fixed validation patients.
6. Select by mean missing-slice PSNR at strides 8/16, requiring non-decreasing mean SSIM versus corrected reference. Compare on the ten historical benchmark patients. Retain reference if no qualifying improvement.
7. Update README; prepare sourced prior-art comparison, labeled/unlabeled diagrams and all 29 sections of a new filled invention disclosure. Flag unverifiable inventor and administrative facts.

## Acceptance criteria

- Normalization statistics never read missing ground-truth intensities; training and inference use identical acquired-only preprocessing.
- Patient sets remain disjoint; acquired targets excluded from diffusion training; acquired slices preserved exactly.
- Invalid indices/shapes/checkpoints fail clearly; seeds and experiment provenance recorded; original outputs preserved.
- Single/multiple modality exports retain axis order; missing patient geometry explicitly documented.
- Tests and smoke checks pass. Results include per-patient metrics, variability, timing and checkpoint identities; no promised performance improvement.
- Disclosure retains template table structure, addresses every section, distinguishes measured findings from proposed uses and cites factual sources.

## Initial findings

- Training uses target-derived intensity statistics; inference normalizes full ground truth before sparsification.
- Diffusion includes acquired targets (context can expose target); reconstruction infers missing indices from stride instead of acquired indices.
- Absent checkpoints silently leave untrained/frozen models in use; inference does not load saved EMA.
- Named B-spline baseline is linear interpolation followed by Gaussian smoothing.
- Multi-modal NIfTI transpose fails; identity affine does not establish patient geometry.
- Historical results: ten patients, 200 records; diffusion exceeds linear PSNR at strides 8/16 but underperforms at 2/4.
- Presentation claims of scan acceleration, clinical efficacy and guaranteed coherence lack supporting validation.

## Defaults

Local compute only. No paid cloud service, filing or publication. Preserve source form/checkpoints/results. Unverified identities, ownership, funding and public disclosure history require inventor confirmation.

## Completed review outcome

- Reliability corrections and focused regression checks implemented; all 34 tests pass.
- Training and explicit resume smoke checks pass; a saved selected reconstruction repeats exactly with its seed (maximum absolute difference 0).
- Bounded protocol completed in 40.36 minutes: reference 10.67, fine-tuning 18.49, selection/final comparisons 11.21 minutes. Fine-tuning stopped at its 3,000-iteration cap.
- No fine-tuned setting met the PSNR/SSIM selection rule; the corrected original raw 25-step smoothed reference remains selected. Original checkpoints and historical results are preserved.
- README, findings, inventory/data audit, per-patient comparisons, validation grid, source identities and disclosure are saved separately.
- The new disclosure populates all 29 sections, retains the template tables and includes four drawings and 11 source-register entries. Inventor facts require confirmation.
- Word's background COM server failed to initialize, preventing native page rendering. Table structure and embedded drawings were checked directly; final pagination needs a check in Word.
