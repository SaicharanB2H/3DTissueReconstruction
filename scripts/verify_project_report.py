"""Audit the report artifact without changing the Word document."""
from pathlib import Path
import sys, json, hashlib, zipfile, csv, statistics
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.review_tools'))
from lxml import etree
from docx import Document
OUT = ROOT / 'results/review/project_report'
path = ROOT / 'BCSE497J_Project_Report_23BAI1536.docx'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
audit = json.loads((OUT / 'build_audit.json').read_text())
pages = json.loads((OUT / 'report_final_page_text.json').read_text(encoding='utf-8'))
doc = Document(path)
checks = {}
checks['template_preserved'] = digest(Path(r'C:\Users\saich\Downloads\BCE497J PROJECT REPORT TEMPLATE.docx')) == digest(OUT / 'Report_Template_Preserved.docx') == audit['template_sha256']
checks['source_snapshots_match'] = all(digest(ROOT / x['file']) == x['sha256'] for x in audit['source_snapshots'])
checks['page_count_in_range'] = 60 <= len(pages) <= 100
checks['continuous_arabic_footers'] = all(x['text'].strip().splitlines()[-1] == str(x['page']) for x in pages)
names = ['Gudikandula Sai Charan Chary', 'Ayush Chaurasia', 'Chahanna Nirov Sheth']
checks['team_on_first_two_pages'] = all(all(n.casefold() in x['text'].casefold() for n in names) and all(reg in x['text'] for reg in ['23BAI1536','23BAI1543','23BAI1505']) for x in pages[:2])
checks['individual_submission_pages'] = all(names[0] in pages[i]['text'] and all(n not in pages[i]['text'] for n in names[1:]) for i in [2, 3, 5])
text = '\n'.join(x['text'] for x in pages)
checks['external_guide_certificate_omitted'] = 'EXTERNAL GUIDE' not in text.upper()
checks['no_word_field_errors'] = 'Error! Reference' not in text and 'Error! Bookmark' not in text
checks['correct_submission'] = all('October 2026' in x['text'] for x in pages[:2])
checks['tables_count'] = len(doc.tables) == 19
with zipfile.ZipFile(path) as z:
    xml = etree.fromstring(z.read('word/document.xml'))
    ns = {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    checks['native_equations_count'] = len(xml.findall('.//m:oMath', ns)) == 13
    media = {hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
    checks['approved_drawings_unchanged'] = all(digest(ROOT / p) in media for p in ['results/review/drawings/architecture_labeled.png', 'results/review/drawings/operation_labeled.png'])
s = doc.sections[0]
geometry = {k: round(getattr(s, k).cm, 3) for k in ['page_width', 'page_height', 'left_margin', 'right_margin', 'top_margin', 'bottom_margin']}
checks['page_geometry'] = all(abs(geometry[k] - v) < .02 for k, v in {'page_width':21, 'page_height':29.7, 'left_margin':3.81, 'right_margin':2.54, 'top_margin':2.54, 'bottom_margin':2.54}.items())
rows = list(csv.DictReader((ROOT / 'results/review/test_reference.csv').open()))
metrics = []
for stride in [8,16]:
    for method in ['conditional_diffusion','linear']:
        group = [r for r in rows if int(r['stride']) == stride and r['method'] == method]
        result = {'stride':stride,'method':method,'patients':len(group)}
        for key, precision in [('psnr_missing',3),('ssim_missing',4)]:
            values = [float(r[key]) for r in group]
            result[key] = {'mean':statistics.mean(values),'sample_std':statistics.stdev(values)}
            checks[f'metric_{stride}_{method}_{key}'] = f'{statistics.mean(values):.{precision}f}' in text
        metrics.append(result)
result = {'artifact':str(path),'sha256':digest(path),'pages':len(pages),'renderer':'Microsoft Word native PDF export; PDFium rasterization at 108 dpi','visual_review':{'pages_reviewed':list(range(1,87)),'result':'Readable; no clipping or overlapping content observed. Natural wrapping retained in code listings.'},'geometry_cm':geometry,'checks':checks,'measured_summary':metrics,'passed':all(checks.values())}
(OUT / 'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['passed']:
    raise SystemExit('Report audit failed')
