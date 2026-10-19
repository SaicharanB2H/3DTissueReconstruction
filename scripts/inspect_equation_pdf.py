import sys,json
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.review_tools'))
import pypdfium2 as pdfium
import pdfplumber
OUT=ROOT/'results/review/disclosure_equations'
input=Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'Invention_Disclosure_Final_Equations.pdf'
label=sys.argv[2] if len(sys.argv)>2 else 'lo'
pdf=pdfium.PdfDocument(str(input));render=OUT/(label+'_pages');render.mkdir(exist_ok=True)
pages=[]
with pdfplumber.open(input) as source:
    for i,page in enumerate(pdf):
        page.render(scale=1.5).to_pil().save(render/f'page-{i+1:02d}.png')
        text=source.pages[i].extract_text() or ''
        pages.append({'page':i+1,'text':text})
        print(i+1,len(text),text[:90].replace('\n',' | '))
(OUT/(label+'_page_text.json')).write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
