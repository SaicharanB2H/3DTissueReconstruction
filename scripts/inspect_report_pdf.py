from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.review_tools'))
sys.stdout.reconfigure(encoding='utf-8')
import pypdfium2 as pdfium
import pdfplumber
OUT=ROOT/'results/review/project_report'
path=Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'report_native.pdf'
label=sys.argv[2] if len(sys.argv)>2 else 'report'
pages=[];output=OUT/(label+'_pages');output.mkdir(exist_ok=True)
pdf=pdfium.PdfDocument(path)
with pdfplumber.open(path) as source:
 for i,page in enumerate(pdf):
  page.render(scale=1.5).to_pil().save(output/f'page-{i+1:03d}.png')
  t=source.pages[i].extract_text() or ''
  pages.append({'page':i+1,'text':t,'words':len(t.split())})
  print(i+1,len(t),t[:80].replace('\n',' | '))
(OUT/(label+'_page_text.json')).write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
