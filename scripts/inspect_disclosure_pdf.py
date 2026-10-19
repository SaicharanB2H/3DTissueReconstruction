"""Render the native Word PDF and retain page text for layout verification."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.review_tools'))
import pypdfium2 as pdfium
import pdfplumber
OUT=ROOT/'results/review/disclosure_final'
pdf=pdfium.PdfDocument(str(OUT/'native_word.pdf'))
pages=[]
render=OUT/'native_pages';render.mkdir(exist_ok=True)
with pdfplumber.open(OUT/'native_word.pdf') as textpdf:
    for i,page in enumerate(pdf):
        page.render(scale=2).to_pil().save(render/f'page-{i+1:02d}.png')
        text=textpdf.pages[i].extract_text() or ''
        pages.append({'page':i+1,'text':text,'width':page.get_width(),'height':page.get_height()})
        print(i+1,len(text),text[:105].replace('\n',' | '))
(OUT/'native_page_text.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False),encoding='utf-8')
