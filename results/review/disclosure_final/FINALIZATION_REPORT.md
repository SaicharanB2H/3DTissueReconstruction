# Disclosure finalization report

**Finalized:** 9 October 2026  
**Deliverable:** `Invention_Disclosure_Final.docx` in the project root  
**Source:** `Invention_Disclosure_Revised_v2.docx`  
**Status:** Editorial and layout review complete. Unconfirmed factual fields remain explicitly flagged.

## Changes

- Fixed Q25's floating market table and narrow text column. The table is inline, uses balanced widths, and distinguishes company revenue/forecast context from the prototype's unestablished addressable market and adoption.
- Reused all four approved diagram images without changing their bytes. Set A contains labeled architecture and operation flow; Set B contains clean copies. Q12 of the original form expressly requests both sets, including flowcharts. Clean drawings intentionally contain no internal text or numbers.
- Kept each figure caption with its image using inline grouping rows, because paragraph page breaks were ignored inside the form in Word compatibility mode. Updated Q13 to match the final Figure 1–4 order and reference numerals.
- Preserved the outer 67-row form and its 10-column grid. Merged the explanatory answer below Q28's nine-level TRL grid across the full form width; retained the grid and provisional TRL 4 assessment.
- Kept the prior-art and parameter tables intact on individual pages. Standardized body/source fonts and spacing, added diagram alt text and page numbers, and removed inherited unconfirmed author metadata from the final copy.
- Clarified that reported standard deviations are across ten per-volume metric values. PSNR pools missing-plane pixel errors within each volume; SSIM averages 2D slice/channel values. Recorded the 100 dB near-zero-MSE sentinel, separate VAE/diffusion noise draws, degenerate normalization fallback and smoothed-linear baseline terminology.
- Added the original DDIM reference (R20). Recorded Philips' SEC filing date as 19 February 2026. Preserved the measured/proposed distinction and the unsuccessful fine-tuning outcome.
- Reduced repeated qualifications using cross-references while retaining the limitations, attribution history and open confirmations.

## Evidence and references

The final source register contains R1–R20. The final machine-readable copy is `sources.json` in this folder; the prior revision's source records remain preserved. Company address/product and market context checks used official manufacturer pages, Philips' 2025 results, its February 2026 Diagnosis & Treatment presentation and SEC filing. The DDIM citation uses the authors' arXiv record.

Philips' EUR 17.8 billion group sales and EUR 8.5 billion Diagnosis & Treatment sales describe 2025 company revenues. Its 2026–2028 mid-single-digit comparable-sales CAGR is a company segment forecast. None is presented as the market size or demand for this invention.

Prior-art overlap remains substantial. R5's full description and eleven claims were retrieved in the preceding revision; English is machine translated. Official-register family/status verification, original Korean/formula review, qualified patent assessment and the relevant inventor chronology remain pending. No novelty, non-infringement, clinical efficacy or commercial-interest conclusion was added.

## Verification

Microsoft Word opened the final DOCX and exported `native_word_final.pdf` through its native UI. All **18 pages** were rendered to 144-DPI PNGs and visually inspected at their original resolution. No clipping, overlapping text, detached drawing captions, missing diagrams or unreadable table layout remained.

| Item | Final Word page | Outcome |
|---|---:|---|
| Labeled architecture, Figure 1 | 4 | Complete and readable |
| Labeled operation flow, Figure 2 | 5 | Complete and readable |
| Clean architecture, Figure 3 | 6 | Complete and intentionally unlabeled |
| Clean operation flow, Figure 4 | 7 | Complete and intentionally unlabeled |
| Prior-art comparison table | 12 | Entire table on one page |
| Workable-parameter evidence table | 13 | Entire table on one page |
| Q25 market text and table | 14 | Inline, readable, no overlap |
| Q28 TRL grid and explanation | 15 | Full-width explanation; grid retained |
| Source register R1–R20 | 17–18 | Readable references and URLs |

Structural checks confirmed all **29 answers**, four unchanged embedded figure images, 67 outer rows, no floating tables, a 12-word title, and no replacement-character/mojibake artifacts. The saved v2 source hash remains unchanged.

Final DOCX SHA-256: `3b769c1386f9b07b41c3b180b35b6d4ce8a6eb6e4c70ba7d70b19f0bbbf1ee9a`.

`verification.json` records the document/PDF hashes, image hashes, checked page numbers and outcomes. `native_render_status.json` records the successful native export. Earlier hidden Word COM export stalled; the later native UI export supersedes that failure. A checksum/signature-verified project-local LibreOffice fallback was administratively extracted and exported successfully; native Word is the acceptance evidence. No system-wide renderer installation was performed. PDFs and page PNGs are internal QA records, not additional requested deliverables.

## Preserved files and remaining confirmations

The original template, earlier filled/revised documents, v2, approved drawing assets, checkpoints, historical results and project source evidence remain preserved. A v2 backup is retained in this folder. No new model experiments, external submission, publication or outreach were performed during this finalization.

Directly confirmed: **Dr. Sarita Singh**, **SCOPE Department, Vellore Institute of Technology, Chennai**. Individual technical contributions, full inventor list/order, required contact details, applicant/ownership, institutional rights/funding, conception/public-disclosure dates, data/converter/checkpoint provenance and institutional TRL approval still require inventor confirmation. Names from presentations remain candidates. The final document explicitly identifies these unknowns; editorial completion does not resolve them.
