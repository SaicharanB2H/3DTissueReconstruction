# Prior-art revision record

Search/access date: **8 October 2026**. This is a preliminary technical comparison; novelty and inventive step remain unestablished.

## R5 retrieval and claim review

The earlier full-text retrieval limitation was resolved. The full published description and **claims 1–11** of [WO2025264018A1](https://patents.google.com/patent/WO2025264018A1/en) were saved as `R5_retrieval.html`; extracted claims are in `R5_claims.json`.

The page contains Korean text with English machine translation. Claims 1, 10 and 11 concern diffusion-based conversion of volume data using related slices trained in bundles, respectively as method, apparatus and stored instructions. Dependent claims address prediction averaging, adjacent-slice training, objectives and correction. The implemented system has meaningful conceptual overlap. Acquired-only calibration, explicit depth/offset/gap conditioning and acquired-plane restoration are possible implementation distinctions requiring assessment; they are not established novelty or non-infringement conclusions. The project has no implemented manifold correction. Formula images and translation precision require original-publication review.

Google's index reports priority **20 June 2024**, PCT filing **19 June 2025**, publication **26 December 2025**, and “Pending” status. The [KR102905764B1](https://patents.google.com/patent/KR102905764B1/en) entry supplies a related Korean grant lead. These index entries are not independently verified legal conclusions; complete family and current status remain pending.

## Wider screening

| Reference | Primary source | Technical relevance / limitation |
|---|---|---|
| R1 | [Rombach et al., latent diffusion](https://arxiv.org/abs/2112.10752) | Latent denoising and cross-attention are established. |
| R2 | [Bubeck et al., CaLID](https://arxiv.org/abs/2508.13826) | Sparse MRI latent interpolation is a close conceptual overlap; brain organ choice alone does not establish inventive step. |
| R3 | [Du et al., ISCS](https://arxiv.org/abs/2602.04162) | Controls inter-slice stochasticity; the project uses independent seeded noise with optional depth smoothing. |
| R4 | [US11410271B2](https://patents.google.com/patent/US11410271B2/en) | Missing-slice interpolation; claims 1, 11, 14 reviewed. Indexed US continuation relationship to US11024004B2 requires official verification. |
| R17 | [Wang et al., thick-slice lumbar MRI](https://arxiv.org/abs/2412.00511) | Abstract-level screening identifies a latent diffusion prior for vertebral reconstruction; detailed claim-equivalent comparison remains pending. |
| R18 | [Tang et al., LDPM](https://arxiv.org/abs/2411.02951) | Abstract-level screening identifies MRI VAE/latent diffusion for undersampled reconstruction; the project does not implement k-space consistency. |

No head-to-head experiment against these systems was performed. Their reported clinical, segmentation or speed outcomes do not transfer to this project.

## Search audit

- Databases: arXiv and Google Patents; manufacturer and dataset primary sources for background. WIPO PATENTSCOPE and Espacenet links attempted but unavailable through the browsing tool.
- Queries: `WO2025264018A1 claims inventor priority`; `patent sparse slice MRI latent diffusion reconstruction prior art`; `latent interpolation cardiac diffusion`; `thick slice MRI latent diffusion`; `undersampled MRI latent diffusion prior`.
- Patent jurisdictions/families screened: United States, PCT/WO and Korea. This is not a complete jurisdiction/family search.
- Official-register verification, prosecution histories, translation/formulas, original human contribution and chronology, and a qualified patent assessment remain pending.
- Complete bibliography and access/version metadata are in `sources.json` and the revised DOCX. Undated manufacturer pages are identified as undated.
