"""Revise the saved disclosure in place structurally, writing a separate DOCX.

Use only confirmed user facts and the recorded project evidence. The input
document, original template, original answers, results and checkpoints are kept.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.review_tools'))
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

REVIEW = ROOT / 'results/review'
OUT = REVIEW / 'disclosure_revision'
OUT.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / 'Invention_Disclosure_Filled.docx'
TARGET = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'Invention_Disclosure_Revised.docx')
ROWS = {1:1,2:3,3:5,4:7,5:10,6:12,7:14,8:16,9:18,10:20,11:22,
        12:24,13:26,14:28,15:30,16:32,17:34,18:36,19:38,20:41,21:43,
        22:45,23:47,24:49,25:52,26:54,27:56,28:64,29:66}
SCOPE = 'retrospective sparse-slice FLAIR MRI reconstruction research prototype'
PENDING = 'Requires inventor confirmation.'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def clean(text):
    # Repair known mixed-encoding sequences in the prior draft only.
    return (text.replace('â€™', "'").replace('â€“', '-').replace('â€”', '-'))

def paragraph(cell, text):
    p = cell.add_paragraph(text)
    p.paragraph_format.space_after = Pt(5)
    for r in p.runs:
        r.font.size = Pt(10)
    return p

def replace_answer(cell, text):
    # Keep nested tables and drawing paragraphs, including their relationships.
    for p in list(cell.paragraphs):
        if not p._p.xpath('.//w:drawing'):
            p._p.getparent().remove(p._p)
    for line in text.split('\n'):
        paragraph(cell, line)

def table_style(table):
    table.autofit = False
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    header = OxmlElement('w:tblHeader')
    table.rows[0]._tr.get_or_add_trPr().append(header)

def add_table(cell, headers, records, widths):
    t = cell.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    for c, value, width in zip(t.rows[0].cells, headers, widths):
        c.text = value
        c.width = Inches(width)
        for r in c.paragraphs[0].runs:
            r.bold = True
    for record in records:
        for c, value, width in zip(t.add_row().cells, record, widths):
            c.text = value
            c.width = Inches(width)
    table_style(t)
    return t

def sources():
    records = read_json(REVIEW / 'sources.json')
    additions = [
      ('R12', 'GE HealthCare Technologies Inc.', 'Investor FAQs',
       'Undated official page; accessed 8 October 2026',
       'https://investor.gehealthcare.com/resources/investor-faqs'),
      ('R13', 'GE HealthCare', 'AIR Recon DL receives FDA clearance of 3D and motion-insensitive imaging applications',
       '29 September 2022, manufacturer announcement',
       'https://www.gehealthcare.com/en/about/newsroom/press-releases/air-recon-dl-receives-fda-clearance-of-3d-and-motion-insensitive-imaging-application'),
      ('R14', 'Canon Medical Systems Corporation', 'Corporate Information',
       'Undated official corporate page; accessed 8 October 2026',
       'https://sg.medical.canon/about-us/corporate/corporate-information/'),
      ('R15', 'Garry E. Gold; Canon Medical Systems', 'Deep Learning Reconstruction in Clinical MRI: a new realm combining ultra-high resolution and increased productivity',
       'Undated manufacturer publication; accessed 8 October 2026',
       'https://global.medical.canon/publications/mri/ai/deep-learning-reconstruction-in-clinical-mri/'),
      ('R16', 'FUJIFILM Healthcare Americas Corporation', 'Echelon Synergy MRI',
       'Undated official product/contact page; accessed 8 October 2026',
       'https://healthcaresolutions-us.fujifilm.com/products/diagnostic-imaging/mri/echelon-synergy/'),
      ('R17', 'Yanke Wang; Yolanne Y. R. Lee; Aurelio Dolfini; Markus Reischl; Ender Konukoglu; Kyriakos Flouris',
       'Energy-Based Prior Latent Space Diffusion model for Reconstruction of Lumbar Vertebrae from Thick Slice MRI',
       'arXiv:2412.00511v1, 30 November 2024; MICCAI DGM workshop 2024',
       'https://arxiv.org/abs/2412.00511'),
      ('R18', 'Xingjian Tang; Jingwei Guan; Linge Li; Ran Shi; Youmei Zhang; Mengye Lyu; Li Yan',
       'LDPM: Towards undersampled MRI reconstruction with MR-VAE and Latent Diffusion Prior',
       'arXiv:2411.02951; v1 5 November 2024; v3 6 June 2025; EMBC 2025',
       'https://arxiv.org/abs/2411.02951'),
      ('R19', 'Newcure M Inc.; named inventors in the Korean publication',
       'KR102905764B1: Method, apparatus and computer program for 3D volume data conversion using multi-slice learned 2D diffusion models',
       'Grant publication 31 December 2025; indexed family/status only, official-register verification pending',
       'https://patents.google.com/patent/KR102905764B1/en')]
    metadata = {
      'R1': ('Robin Rombach; Andreas Blattmann; Dominik Lorenz; Patrick Esser; Björn Ommer',
             'High-Resolution Image Synthesis with Latent Diffusion Models',
             'arXiv:2112.10752; v1 20 December 2021; v2 13 April 2022; CVPR 2022'),
      'R2': ('Niklas Bubeck; Suprosanna Shit; Chen Chen; Can Zhao; Pengfei Guo; Dong Yang; Georg Zitzlsberger; Daguang Xu; Bernhard Kainz; Daniel Rueckert; Jiazhen Pan',
             'Latent Interpolation Learning Using Diffusion Models for Cardiac Volume Reconstruction',
             'arXiv:2508.13826; v1 19 August 2025; v4 24 July 2026'),
      'R3': ('Chenhe Du; Qing Wu; Xuanyu Tian; Jingyi Yu; Hongjiang Wei; Yuyao Zhang',
             'Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity',
             'arXiv:2602.04162; v1 4 February 2026; v2 9 February 2026; ICLR 2026'),
      'R4': ('Maria H. Oleszkiewicz; Blazej R. Rutkowski; Wojciech Sobala; Piotr Padkowski; International Business Machines Corporation',
             'US11410271B2: Reconstructing missing slices in slice-based scans of 3D objects',
             'Grant publication 9 August 2022; indexed priority 31 August 2018; claims 1, 11, 14 reviewed'),
      'R5': ('Newcure M Inc.; publication names inventors 추교빈 and 전영준',
             'WO2025264018A1: 3D volume data conversion method and device using multi-slice-trained 2D diffusion model, and computer program',
             'Published 26 December 2025; PCT/KR2025/008522; full description and claims 1-11 retrieved; English machine translation'),
      'R6': ('CBICA, University of Pennsylvania; BraTS 2020 organizers',
             'Multimodal Brain Tumor Segmentation Challenge 2020: Data', '2020 challenge data page; page update date not stated'),
      'R7': ('Siemens Healthcare GmbH', 'Value Partnerships - Contact Us', 'Undated official contact page'),
      'R8': ('Siemens Healthineers', 'Deep Resolve', 'Undated official manufacturer product page'),
      'R9': ('Koninklijke Philips N.V.', 'Annual Report 2025, Form 20-F: Cover Page / Document and Entity Information', 'Fiscal year ended 31 December 2025; 2026 SEC filing'),
      'R10': ('Koninklijke Philips N.V.', 'Philips 2025 annual results', '2025 annual results, released in 2026'),
      'R11': ('Bert van Meurs; Jie Xue; Koninklijke Philips N.V.', 'Capital Markets Day: Diagnosis & Treatment', '10 February 2026, slides 3 and 5')
    }
    result = []
    for code, _, url in records:
        organization, title, date = metadata[code]
        result.append(dict(id=code, authors_or_organization=organization, title=title,
                           publication_date_or_version=date, url=url, accessed='2026-10-08'))
    for code, org, title, date, url in additions:
        result.append(dict(id=code, authors_or_organization=org, title=title,
                           publication_date_or_version=date, url=url, accessed='2026-10-08'))
    return result

def revise():
    backup = OUT / 'Invention_Disclosure_Before_Revision.docx'
    if not backup.exists():
        shutil.copy2(INPUT, backup)
    before_sha = digest(INPUT)
    doc = Document(INPUT)
    original = read_json(REVIEW / 'disclosure_sections.json')
    content = {int(k): clean(v) for k, v in original.items()}
    # Stop rather than silently discard saved edits since the previous fill.
    changed = [n for n,row in ROWS.items() if original[str(n)].strip() not in doc.tables[0].rows[row].cells[1].text.strip()]
    if changed:
        raise RuntimeError(f'Input answers changed by the user in sections {changed}; reconcile before rebuilding.')
    facts = read_json(OUT / 'inventor_confirmations.json')
    tradeoff = ('The selected configuration is the corrected original raw 25-step reference; the fine-tuned candidate did not qualify and no gain over that reference is demonstrated. '
                'At stride 8, diffusion has higher PSNR than linear interpolation (28.037 versus 27.628 dB), while linear interpolation has higher SSIM (0.9226 versus 0.9054): an explicit PSNR/SSIM trade-off. '
                'At stride 16, diffusion has higher mean PSNR and SSIM in this internal comparison. '
                'Reported +/- values are sample standard deviations across ten volume-level means, not confidence intervals. '
                'PSNR and SSIM are two-dimensional image-fidelity measures including background; they do not establish lesion preservation, topology, diagnosis or clinical utility.')
    preservation = ('Exact preservation applies to the supplied normalized acquired planes restored after optional smoothing. '
                    'Percentile clipping is not invertible preservation of raw scanner intensities, and preservation of acquired planes is not evidence that synthesized planes are anatomically correct.')
    chronology = ('Earliest conception date and dated supporting record: '+PENDING+' Earliest working implementation/reduction-to-practice date and record: '+PENDING+' '
                  'Public repository, thesis deposit, slides, seminar/defense, poster, abstract, demonstration or other disclosure: dates, audience/recipients, URLs and confidentiality terms '+PENDING+' '
                  'Local file timestamps and presentation titles do not establish conception or public disclosure. The 8 October 2026 review date is not asserted as the invention date.')
    attribution = ('Attribution record: pre-review repository evidence shows the 2D variational autoencoder, shared residual encoder, Fourier depth/offset/gap inputs, hybrid conditioning and latent diffusion pipeline. '
                   'The 8 October 2026 Codex-assisted review added or corrected acquired-only normalization, exclusion of acquired diffusion targets, supplied-index handling, checkpoint/configuration validation, resume/EMA handling, baseline naming, geometry/export metadata and reproducible bounded experiments. '
                   'Review-generated claim wording and experiment reports are drafting/evaluation assistance. Which features each human conceived, and when, '+PENDING+' This implementation history does not determine legal inventorship.')
    provenance = ('Provenance to be supplied: exact BraTS release/download or redistribution source and URL, acquisition date, applicable access/license/redistribution terms; HDF5 converter source/version and channel order, axis orientation, intensity transformations and mask-label mapping; original VAE/diffusion training and checkpoint creation records, licenses and any external weights/backbones/data. '+PENDING+' '
                  'The local directory name suggests BraTS 2020; R6 describes that challenge, but does not authenticate this local conversion or establish its rights. Stored HDF5 metadata does not supply physical spacing/orientation. '
                  'Source code uses custom model classes and the repository stores trained weights; these facts alone cannot prove that every original training input/weight was locally created. The measured learned model is single-channel, assumed FLAIR; four-channel I/O verification does not demonstrate joint multi-contrast learning.')
    content[2] = ('Demonstrated scope: '+SCOPE+'. MRI means magnetic resonance imaging; FLAIR means fluid-attenuated inversion recovery. '
                  'Problem: sparse axial sampling leaves missing planes. Solution: bounding acquired slices and depth coordinates condition a two-dimensional latent denoising model, whose generated planes are assembled with the acquired planes. '
                  'VAE means variational autoencoder; DDIM means denoising diffusion implicit model. Latent diffusion and cross-attention are established (R1); related sparse MRI reconstruction is documented in R2. '
                  'Scanner acceleration and diagnostic use are proposed applications requiring separate evidence.')
    content[3] = content[3].replace('DDIM;', 'denoising diffusion implicit model (DDIM);')
    content[3] += '\nPSNR: peak signal-to-noise ratio. SSIM: structural similarity index measure. MSE: mean squared error. EMA: exponential moving average. HDF5: Hierarchical Data Format version 5. NIfTI: Neuroimaging Informatics Technology Initiative. TRL: technology readiness level.'
    content[4] += '\nThe implemented learned models are two-dimensional; stacking generated planes creates a volume. No three-dimensional VAE/U-Net or CNN/ViT backbone choice is implemented.'
    content[5] = ('R1 establishes latent diffusion and cross-attention; R2 addresses sparse cardiac MRI reconstruction and is a close conceptual overlap; R3 addresses inter-slice noise correlation; R4 discloses recursive missing-slice interpolation. '
                  'R5 full description and claims 1-11 were retrieved during this revision; section 21 provides the comparison and retrieval limitations. '
                  'Additional screening: R17 addresses thick-slice lumbar reconstruction with a latent diffusion prior; R18 addresses undersampled MRI with an MRI VAE and latent diffusion prior. '
                  'These applications reinforce that latent diffusion in medical reconstruction is established. No head-to-head performance comparison was conducted.')
    content[6] += '\nR5 and R17/R18 further narrow the basis for broad novelty claims. Chronology and original human contributions remain unconfirmed.'
    content[7] = content[7].replace('HDF5 slice reader', 'Hierarchical Data Format version 5 (HDF5, stored as .h5 files) slice reader').replace('NIfTI/HDF5 exporter', 'Neuroimaging Informatics Technology Initiative (NIfTI)/HDF5 exporter')
    content[8] += '\n'+preservation
    content[9] += '\n'+chronology+'\nThe R5 claim comparison appears in section 21. Claim applicability and priority require specialist assessment using confirmed dates and official records.'
    for n in (10,14,15,17,19,23):
        content[n] += '\n'+tradeoff
    content[12] += '\nFigures 1-4 represent the implemented 2D pipeline. Drawing paragraphs are kept together; labels and reference numerals are matched to section 13. Native pagination verification is recorded in the revision verification attachment; any unavailable rendering remains explicitly pending.'
    content[13] += '\n201 validates supplied indices, including both boundaries; 202 uses acquired-only statistics; 203 finds bounding contexts; 204 applies the 2D encoder/U-Net and frozen VAE decoder; 205 optionally smooths then restores normalized acquired values; 206 exports voxel-index geometry unless verified geometry is supplied.'
    content[14] += '\n'+preservation+'\n'+provenance
    content[15] += ('\nAudit trail: results/review/test_reference.csv and test_selected.csv contain per-volume metrics; test_reference.json and test_selected.json contain settings and checkpoint identities; '
                    'experiment_manifest.json, validation_reference.csv, validation_candidate.csv, selection.json, paired_comparison.csv and completion.json record the protocol and selection. '
                    'Global seed=42; reconstruction seed=42+100*volume_id+stride, e.g. volume 14/stride 8 uses seed 1450. '
                    'Evaluation uses range [-1,1], missing planes only, 11x11 Gaussian SSIM window sigma=1.5 with zero padding. Generation timing excludes smoothing/metrics, while baseline row timing includes metrics; these values do not support a fair speed-superiority comparison.')
    ref = read_json(REVIEW / 'test_reference.json')
    content[15] += '\nSelected/reference diffusion SHA-256: '+ref['checkpoint']['sha256']+'\nFrozen VAE SHA-256: '+ref['vae_checkpoint']['sha256']
    content[16] += '\nEMA means exponential moving average of weights. Configuration acceptance is not performance validation. '+provenance
    content[17] = content[17].replace('No external data or third-party clinical study is used.', 'The review experiments use the local collection; no independent clinical study was supplied. Earlier checkpoint/data provenance remains unverified.')
    content[17] += '\n'+provenance+'\nFilename IDs are treated as patient-volume identifiers; linkage to original patient identities and converter lineage '+PENDING
    content[18] = chronology
    content[19] = 'Development stage: '+SCOPE+'. '+content[19]+'\n'+preservation
    content[20] += '\n'+attribution+'\nClaim concepts are technical drafting proposals, with novelty and inventive step unestablished. Claims 1, 10 and 11 of R5 warrant particular comparison; see section 21.'
    content[21] = ('Search date: 8 October 2026. Databases consulted: arXiv and Google Patents; primary manufacturer/data pages support background facts. '
                   'Queries included "WO2025264018A1 claims inventor priority", "patent sparse slice MRI latent diffusion reconstruction prior art", "latent interpolation cardiac diffusion", "thick slice MRI latent diffusion" and "undersampled MRI latent diffusion prior". '
                   'Jurisdictions/family leads screened: US for R4; PCT/WO and Korea for R5/R19. WIPO PATENTSCOPE and Espacenet links were attempted but not accessible; official-register status/family verification remains pending. '
                   'R5 full HTML description and all 11 claims are saved under results/review/disclosure_revision; English text is a machine translation of Korean, and formula images require original-publication review. '
                   'R5 claims 1, 10 and 11 cover volume conversion using bundled related slices and a trained diffusion model, as method, device and medium. Claims 2-9 add prediction averaging, adjacent-slice training and optional correction steps. '
                   'Overlap: learned slice context and diffusion-based volume assembly. Possible implementation distinctions include acquired-only calibration, explicit depth/offset/gap inputs and exact restoration; their absence from selected claim text does not establish novelty or non-infringement. '
                   'Google indexes priority 20 June 2024, filing 19 June 2025, publication 26 December 2025 and status "Pending"; R19 is an indexed Korean grant lead published 31 December 2025. These are unverified index reports, not legal conclusions. '
                   'R4 indexes a US continuation related to US11024004B2; current enforceability and family scope require official records. This technical search is not exhaustive; qualified patent review remains pending.')
    content[22] += '\nR5 and R17/R18 require further consideration. '+attribution
    content[23] = ('Parameter evidence is separated in the table below. Tested configurations do not establish a continuous effective range. '
                   'Further workable ranges and any earlier supporting experiments '+PENDING+'\n'+tradeoff)
    content[24] = ('Five prospective companies are listed below with official address sources. Interest, outreach and established relationships are unconfirmed for all five. '
                   'Commercial rationale is an inference from their documented MRI/AI reconstruction portfolios, not evidence that they want this invention.\n'
                   '1. Siemens Healthcare GmbH: Henkestr. 127, 91052 Erlangen, Germany (R7). MRI reconstruction expertise through Deep Resolve (R8) makes research/licensing exploration plausible.\n'
                   '2. Koninklijke Philips N.V.: Prinses Irenestraat 59, 1077 WV Amsterdam, Netherlands (R9). Its diagnostic-imaging portfolio and AI investment (R11) support prospective research discussions.\n'
                   '3. GE HealthCare Technologies Inc.: 500 W. Monroe Street, Chicago, Illinois 60661, United States (R12). AIR Recon DL illustrates relevant MRI reconstruction expertise (R13).\n'
                   '4. Canon Medical Systems Corporation: 1385 Shimoishigami, Otawara-shi, Tochigi 324-8550, Japan (R14). AiCE MRI reconstruction provides a related commercial background (R15). The corporate page uses this entity name; contracting entity must be rechecked before outreach.\n'
                   '5. FUJIFILM Healthcare Americas Corporation: 81 Hartwell Avenue, Suite 100, Lexington, Massachusetts 02421, United States (R16). Echelon Synergy/Synergy DLR illustrates MRI reconstruction activity (R16).\n'
                   'Research-oriented positioning: '+SCOPE+' with measured fidelity and preservation of normalized acquired data. Scanner integration, clinical benefit, licensing demand and willingness to pay remain unvalidated. Product claims from R8/R13/R15/R16 do not transfer to this prototype.')
    content[25] = ('A specific addressable market or adoption rate for this prototype has not been established; independent customer research is required. '
                   'Context only: Philips reports EUR 17.8 billion total 2025 sales (R10); its Diagnosis & Treatment presentation reports EUR 8.5 billion segment sales and a company forecast of mid-single-digit comparable-sales CAGR for 2026-2028 (R11). '
                   'These broad company/segment figures are not the market size or revenue potential of sparse-slice software. '
                   'Manufacturer MRI AI products (R8/R13/R15/R16) demonstrate a relevant product category, not demand for generative missing-slice reconstruction or this project. '
                   'Proposed customer research: identify research-only reconstruction workflows and permitted data access, interview imaging labs and manufacturers, measure external lesion-specific fidelity, assess integration effort, and obtain independent evidence for adoption, pricing and willingness to pay. '
                   'No prototype market CAGR, sales forecast, scan savings, diagnostic efficacy or economic benefit is asserted.')
    confirmed = []
    for i, inventor in enumerate(facts['inventors'], 1):
        confirmed.append(f"{i}. {inventor['name']}; affiliation: {inventor.get('affiliation') or PENDING}; technical conception contribution: {inventor.get('contribution') or PENDING}; address/contact: {inventor.get('address') or PENDING}")
    content[26] = ('User-confirmed inventor entries (name/listing supplied by the user; contributions remain separately reviewed):\n'+'\n'.join(confirmed)+'\n'
                   'Complete inventor list, ordering and inventorship assessment: '+('User confirmed the complete list.' if facts['complete_inventor_list_confirmed'] else PENDING)+'\n'
                   'Proposal candidates Gudikandula Sai Charan chary (23BAI1536), Ayush chaurasia (23BAI1543), and Chahanaa Sheth (23BAI1517) remain unconfirmed. '
                   'The earlier guide name Dr. Sarita Kumari is a historical source entry; the user supplied Dr. Sarita Singh. No equivalence or correction of the source presentation is inferred.\n'
                   'Applicant/legal owner, institutional assignment, funding/grants/sponsors, collaborator obligations and third-party rights are separate facts: '+PENDING+' Supervision alone does not establish conception of claimed subject matter.')
    content[27] += '\nDemonstrated scope: '+SCOPE+'.'
    content[28] = ('Technology readiness level (TRL): provisional level 4, based on an integrated software prototype evaluated on retrospective local data in a computing/laboratory setting. '
                   'Institutional definitions and inventor/institution agreement with this assessment: '+PENDING+' '
                   'No relevant clinical environment, qualified product or deployment evidence supports levels 5-9. The template definitions are retained.')
    content[29] = ('Revised on 8 October 2026 from the user-requested change report, saved project evidence and direct user answers. Demonstrated scope: '+SCOPE+'. '
                   'Original template, prior filled disclosure, presentations, checkpoints and historical results remain preserved. '
                   'Inventor answers are recorded verbatim separately; remaining facts are visibly flagged. '+preservation+'\n'+attribution+'\n'
                   'Open confirmations: complete inventor list and individual conception contributions; separate applicant/ownership/funding; conception and reduction-to-practice dates; public disclosure chronology; data/converter/checkpoint provenance and rights; extra effective parameter ranges; institutional TRL assessment. '
                   'Qualified patent review and official-register family/status verification remain pending. No external submission/publication or outreach has been performed.')
    table = doc.tables[0]
    for n,row in ROWS.items():
        replace_answer(table.rows[row].cells[1], content[n])
    # Clean encoding throughout retained prompt/nested-table text without touching drawings.
    for element in doc.element.xpath('.//w:t'):
        if element.text:
            element.text = clean(element.text)
    comparison = table.rows[43].cells[1].tables[0]
    records = [
      ['1','R1: latent diffusion','Latent denoising + cross-attention','MRI bounding-slice/depth implementation; no novelty conclusion'],
      ['2','R2: CaLID','Sparse MRI latent interpolation','Brain FLAIR implementation; conceptual overlap'],
      ['3','R3: ISCS','Inter-slice noise control','Independent seeded noise + smoothing; ISCS not implemented'],
      ['4','R4: US11410271B2','Recursive binary interpolation','Grayscale latent generation; legal scope pending'],
      ['5','R5: WO2025264018A1','Related-slice diffusion; claims 1-11 reviewed','Possible calibration/coordinate/restoration distinctions need review'],
      ['6','R17: thick-slice lumbar MRI','Latent prior for vertebral reconstruction','Different task/output; medical latent-prior concept established'],
      ['7','R18: LDPM','MRI VAE + latent diffusion prior','Undersampled reconstruction; no implemented k-space consistency']]
    for i, record in enumerate(records,1):
        if i >= len(comparison.rows):
            comparison.add_row()
        for c,v in zip(comparison.rows[i].cells, record):
            c.text = v
    table_style(comparison)
    market = table.rows[52].cells[1].tables[0]
    market_records = [
      ['2025 context','Philips total sales EUR 17.8bn (R10); D&T segment EUR 8.5bn (R11); not invention TAM','MRI AI product category documented','R8, R13, R15, R16; no project adoption evidence'],
      ['2026-2028 context','Philips segment forecast mid-single-digit CAGR (R11); no prototype forecast','Customer validation proposed','No established project customers or commercial product'],
      ['Prototype opportunity','Not quantified; independent customer research required','Research-only retrospective FLAIR workflow','Demand, pricing, integration costs and clinical benefit unvalidated']]
    for i,record in enumerate(market_records,1):
        if i >= len(market.rows):
            market.add_row()
        for c,v in zip(market.rows[i].cells, record):
            c.text = v
    table_style(market)
    parameters = [
      ['Sampling strides','4/8/16 equal-probability fine-tuning; corrected quality at 8/16','Other positive strides: proposed, no quality guarantee','training manifest; test CSVs'],
      ['Input dimensions','155 depths; 240x240; learned C=1 assumed FLAIR','Other sizes/modalities/organs need training + evaluation','dataset audit; model config'],
      ['DDIM steps / eta','25/50 compared; selected 25; eta=0','1..1000 accepted by API; effective range unestablished','validation CSVs; selection.json'],
      ['Normalization','Foreground 0.5/99.5 percentiles; [-1,1]; epsilon=1e-7','Other percentiles/ranges: proposed, require experiments','normalization source + focused checks'],
      ['Depth smoothing','Off / Gaussian sigma=0.5 compared; selected on','Other sigma values: proposed; no tested quality range','validation configuration'],
      ['Weights','Raw / available EMA compared; candidate EMA decay=0.999','Other decay values: proposed, unvalidated','checkpoint/configuration records'],
      ['Fine-tuning LR','2e-5; AdamW; weight decay=1e-4; clip norm=1','Other optimizer/LR ranges: unvalidated','finetune manifest and training.csv'],
      ['Batch / budget','Batch 8; 3000 steps; 24000 replacement draws; 18.49min','Other batch/step budgets accepted; quality not established','finetune completion.json'],
      ['Latent / conditioning','4x30x30; scale=0.18215; 17x256 tokens; 8 spatial channels','Changed latent/context widths require retraining','model source/configuration'],
      ['Model selection','Five validation IDs; PSNR mean across 8/16; SSIM must not decrease','Extra validation cohorts/criteria: proposed','validation_reference/candidate.csv'],
      ['Irregular indices / export','Focused checks; both boundaries required; 1/4-channel export','API correctness checks are not quality validation','tests; physical geometry unavailable']]
    add_table(table.rows[47].cells[1], ['Parameter','Demonstrated setting','Proposed / accepted only','Evidence'], parameters, [1,1.65,1.75,1.1])
    # Keep each of the four picture paragraphs intact during pagination.
    drawings_cell = table.rows[24].cells[1]
    picture_paragraphs = [p for p in drawings_cell.paragraphs if p._p.xpath('.//w:drawing')]
    # Put the description ahead of the figures, then caption each complete image.
    labels = ['Figure 1. Labeled architecture (components 101-110)',
              'Figure 2. Unlabeled architecture',
              'Figure 3. Labeled operation flow (steps 201-206)',
              'Figure 4. Unlabeled operation flow']
    for p, label in zip(picture_paragraphs, labels):
        drawings_cell._tc.remove(p._p)
        caption = paragraph(drawings_cell, label)
        caption.paragraph_format.keep_with_next = True
        caption.runs[0].bold = True
        drawings_cell._tc.append(p._p)
        p.paragraph_format.keep_together = True
        p.paragraph_format.keep_with_next = False
    # Replace only the known appended source register; leave the original table.
    remove = False
    for p in list(doc.paragraphs):
        if p.text == 'Source register and evidence attachments':
            remove = True
        if remove:
            p._p.getparent().remove(p._p)
    heading = doc.add_paragraph('Source register and revision evidence')
    heading.runs[0].bold = True
    heading.runs[0].font.size = Pt(14)
    doc.add_paragraph('Access date for all sources: 8 October 2026. Undated pages are identified explicitly; no update date is inferred. Patent index status is not verified official-register status. R5 English text is machine translated.')
    src = sources()
    for item in src:
        doc.add_paragraph(f"{item['id']}. {item['authors_or_organization']}. {item['title']}. {item['publication_date_or_version']}. Accessed {item['accessed']}.\n{item['url']}")
    doc.add_paragraph('Evidence attachments: results/review/disclosure_revision/REVISION_CHANGELOG.md; PENDING_CONFIRMATIONS.md; inventor_confirmations.json; PRIOR_ART_REVISION.md; parameter_evidence.csv; sources.json; disclosure_sections.json; verification.json; per-volume CSVs and checkpoint/configuration JSONs named in section 15. Original review evidence is preserved.')
    doc.core_properties.title = content[1]
    doc.core_properties.subject = SCOPE
    doc.core_properties.comments = 'Revised from the user-requested change report; unconfirmed administrative facts remain flagged.'
    target = TARGET
    try:
        doc.save(target)
    except PermissionError:
        # Word can hold a draft open; preserve it and write the next revision.
        version = 2
        while target.exists():
            target = TARGET.with_name(TARGET.stem + f'_v{version}.docx')
            version += 1
        doc.save(target)
    (OUT/'disclosure_sections.json').write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'sources.json').write_text(json.dumps(src,ensure_ascii=False,indent=2),encoding='utf-8')
    import csv
    with (OUT/'parameter_evidence.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['parameter','demonstrated','proposed_or_accepted_only','evidence'])
        w.writerows(parameters)
    check = Document(target)
    text = '\n'.join(e.text or '' for e in check.element.xpath('.//w:t'))
    cited = set(re.findall(r'\bR\d+\b', '\n'.join(content.values())))
    issue_tokens = [token for token in ('\ufffd','â€™','â€“','â€”') if token in text]
    verification = dict(input_sha256=before_sha,input_preserved=digest(INPUT)==before_sha,
        backup_sha256=digest(backup),revised_path=str(target),revised_sha256=digest(target),
        original_outer_rows=len(doc.tables[0].rows),revised_outer_rows=len(check.tables[0].rows),
        sections_populated=len(content),drawings=len(check.inline_shapes),
        title_words=len(content[1].split()),encoding_issues=issue_tokens,
        bibliography_entries=len(src),uncited_sources=[x['id'] for x in src if x['id'] not in cited],
        native_layout_status='Unavailable: Word sandbox access failed; elevated rendering stalled; no PDF produced; final pagination pending',
        unknown_facts='See PENDING_CONFIRMATIONS.md; not inferred',
        saved_input_changes_detected=changed)
    assert verification['input_preserved'] and verification['sections_populated']==29
    assert verification['drawings']==4 and verification['revised_outer_rows']==67
    assert verification['title_words']<=15 and not issue_tokens and not verification['uncited_sources']
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))

if __name__ == '__main__':
    revise()
