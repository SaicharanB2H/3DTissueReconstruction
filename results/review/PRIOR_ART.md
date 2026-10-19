# Preliminary prior-art comparison

Search date: 8 October 2026. Sources: arXiv, author publications, Google Patents, official BraTS and manufacturer materials. This is a technical comparison, not an exhaustive novelty search or a claim-clearance determination.

| Reference | Existing idea | Relationship to implemented project |
|---|---|---|
| [Rombach et al., latent diffusion, 2021/2022](https://arxiv.org/abs/2112.10752) | Autoencoder latent diffusion and cross-attention conditioning | These general mechanisms are established; the project applies them to neighboring MRI slice conditioning |
| [Bubeck et al., CaLID, first posted 19 August 2025](https://arxiv.org/abs/2508.13826) | Learned latent interpolation for sparse cardiac MRI reconstruction | Strong overlap with the broad purpose and architecture family; brain FLAIR and the particular depth/context implementation do not alone establish inventiveness |
| [Du et al., ISCS, February 2026](https://arxiv.org/abs/2602.04162) | Control diffusion noise across slices to improve consistency | Project uses positional conditioning and optional Gaussian smoothing; correlated trajectory sampling is not implemented |
| [US11410271B2, published 9 August 2022](https://patents.google.com/patent/US11410271B2/en) | Iterative missing-slice insertion using black/white pixel rules; independent claims 1, 11 and 14 inspected | Project uses learned grayscale latent generation at specified depths; both produce intermediate slices and dense stacks |
| [WO2025264018A1](https://patents.google.com/patent/WO2025264018A1/en) | Search-index title identifies 3D conversion using a multi-slice-trained 2D diffusion model | Potentially close patent lead; full text could not be retrieved during this review, so no detailed claim distinction is asserted |

Candidate features to examine together: acquired-only intensity calibration; weight-tied bounding-slice encoding; three-coordinate depth/gap embedding; spatial plus token conditioning; exact acquired-slice preservation. Each needs comparison against full prior-art disclosures and a confirmed earliest disclosure date. No broad monopoly over medical diffusion or missing-slice reconstruction is asserted.

## Data and commercialization sources

- [Official BraTS 2020 data description](https://www.med.upenn.edu/cbica/brats2020/data.html): MRI modalities, expert annotations and original preprocessing. Local file counts and split sizes are measured from this repository.
- [Siemens Healthcare GmbH contact address](https://www.siemens-healthineers.com/en-th/services/value-partnerships/contact-us): Henkestr. 127, 91052 Erlangen, Germany.
- [Siemens Deep Resolve](https://www.siemens-healthineers.com/en-au/magnetic-resonance-imaging/technologies-and-innovations/deep-resolve): an existing commercial AI reconstruction example; no head-to-head comparison was conducted.
- [Philips 2025 filing address](https://www.sec.gov/Archives/edgar/data/313216/000162828026009470/R1.htm): Koninklijke Philips N.V., Prinses Irenestraat 59, 1077 WV Amsterdam, Netherlands.
- [Philips 2025 results](https://www.results.philips.com/ar25): contextual health-technology revenue; not the market size for slice reconstruction.
- [Philips Diagnosis & Treatment investor presentation, 2026](https://www.philips.com/c-dam/corporate/about-philips/investors/calendar/2026/3_-_Diagnosis_and_Treatment.pdf): approximately EUR 8.5 billion segment sales in 2025; company forecast of mid-single-digit comparable sales CAGR in 2026–2028. These figures cover a broad segment, not this invention's addressable market.

Both companies are plausible prospective research or licensing contacts because of their imaging portfolios. Their interest in this project is unconfirmed. A dedicated market-size estimate, adoption rate and revenue forecast require customer research; none is invented here.
