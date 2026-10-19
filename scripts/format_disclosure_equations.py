"""Convert disclosure math to editable OMML through Microsoft's MathML stylesheet."""
from pathlib import Path
import sys, re, copy, zipfile, hashlib, json, shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.review_tools'))
from lxml import etree as E
from docx import Document
from docx.oxml import OxmlElement
from docx.shared import Pt

OUT = ROOT / 'results/review/disclosure_equations'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'Invention_Disclosure_Final.docx'
FILE = ROOT / 'Invention_Disclosure_Final_Equations.docx'
BACKUP = OUT / 'Invention_Disclosure_Before_Equation_Formatting.docx'
if not BACKUP.exists(): shutil.copy2(SOURCE, BACKUP)
doc = Document(BACKUP)
transform = E.XSLT(E.parse(r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'))
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def atom(t, tag='mi'): return f'<{tag}>{t}</{tag}>'
def n(t): return atom(str(t), 'mn')
def op(t): return atom(t, 'mo')
def row(*s): return '<mrow>'+''.join(s)+'</mrow>'
def sub(b,s): return '<msub>'+b+atom(s)+'</msub>'
def sup(b,s): return '<msup>'+b+n(s)+'</msup>'
def frac(a,b): return '<mfrac>'+a+b+'</mfrac>'
def sqrt(a): return '<msqrt>'+a+'</msqrt>'
def bar(a): return '<mover accent="true">'+a+op('¯')+'</mover>'
def hat(a): return '<mover accent="true">'+a+op('ˆ')+'</mover>'
def par(a): return row(op('('),a,op(')'))
def eq(a,b): return row(a,op('='),b)
z=atom('z'); zp=sub(z,'prev'); zn=sub(z,'next'); z0=sub(z,'0'); zt=sub(z,'t')
vmin=sub(atom('v'),'min'); vmax=sub(atom('v'),'max'); eps=atom('ε')
at=sub(bar(atom('α')),'t'); ap=sub(bar(atom('α')),'prev')
et=sub(eps,'θ'); pred=par(row(zt,op(','),atom('t'),op(','),atom('context')))
gap=row(zn,op('−'),zp)
power10=lambda s:sup(n(10),s)
formulas={
106:[('Coordinates: ',eq(atom('α'),frac(row(z,op('−'),zp),gap))),('; depth: ',eq(atom('depth'),frac(z,row(atom('D'),op('−'),n(1))))),('; gap: ',eq(atom('gap'),frac(gap,n(16))))],
107:[('Normalization: ',eq(sub(atom('I'),'norm'),row(n(2),op('·'),frac(row(atom('clip'),par(row(atom('I'),op(','),vmin,op(','),vmax)),op('−'),vmin),row(vmax,op('−'),vmin,op('+'),eps)),op('−'),n(1)))),(', with background mapped to −1 and ',eq(eps,power10('−7')))],
108:[('Clean latent: ',eq(z0,row(n('0.18215'),op('·'),par(row(atom('μ'),op('+'),atom('σ'),op('·'),eps)))))],
109:[('Forward diffusion: ',eq(zt,row(sqrt(at),op('·'),z0,op('+'),sqrt(row(n(1),op('−'),at)),op('·'),eps)))],
110:[('Noise objective: ',row(atom('E'),op('['),sup(row(op('‖'),eps,op('−'),et,pred,op('‖')),2),op(']')))],
111:[('Deterministic DDIM: ',eq(hat(z0),frac(row(zt,op('−'),sqrt(row(n(1),op('−'),at)),op('·'),et),sqrt(at)))),('; ',eq(zp,row(sqrt(ap),op('·'),hat(z0),op('+'),sqrt(row(n(1),op('−'),ap)),op('·'),et)))],
112:[('VAE objective: ',row(sub(atom('L'),'1'),op('+'),n('0.5'),op('·'),atom('MSE'),op('+'),power10('−5'),op('·'),atom('KL'),op('+'),n('0.1'),op('·'),sub(atom('L'),'1, spatial-gradient')))],
113:[('Reconstruction ',eq(atom('MSE'),row(atom('mean'),par(sup(par(row(atom('prediction'),op('−'),atom('target'))),2))))),('; ',eq(atom('MAE'),row(atom('mean'),par(row(op('|'),atom('prediction'),op('−'),atom('target'),op('|')))))),('; ',eq(atom('PSNR'),row(n(10),op('·'),sub(atom('log'),'10'),par(frac(n(4),atom('MSE')))))),(' for [−1, 1], with a 100 dB sentinel returned when ',row(atom('MSE'),op('≤'),power10('−10')))],
114:[('SSIM uses an ',row(n(11),op('×'),n(11))),(' Gaussian window (',eq(atom('σ'),n('1.5'))),('), constants ',sup(par(row(n('0.01'),op('×'),n(2))),2)),(' and ',sup(par(row(n('0.03'),op('×'),n(2))),2))],
}
log=[]
def math(xml):
    result=transform(E.fromstring(('<math xmlns="http://www.w3.org/1998/Math/MathML">'+xml+'</math>').encode()))
    root=copy.deepcopy(result.getroot())
    for r in root.findall('.//{'+M+'}r'):
        pr=E.Element('{'+W+'}rPr'); font=E.SubElement(pr,'{'+W+'}rFonts');font.set('{'+W+'}ascii','Cambria Math');font.set('{'+W+'}hAnsi','Cambria Math')
        sz=E.SubElement(pr,'{'+W+'}sz');sz.set('{'+W+'}val','21');r.insert(0,pr)
    return root
def text_run(p,t,pr=None):
    if not t:return
    r=E.SubElement(p,'{'+W+'}r')
    if pr is not None:r.append(copy.deepcopy(pr))
    tx=E.SubElement(r,'{'+W+'}t');tx.set('{http://www.w3.org/XML/1998/namespace}space','preserve');tx.text=t
def clear(p):
    for child in list(p):
        if child.tag!='{'+W+'}pPr':p.remove(child)
paragraphs=doc._element.xpath('.//w:p')
for i,parts in formulas.items():
    p=paragraphs[i];before=''.join(p.xpath('.//w:t/text()'));pr=p.find('.//{'+W+'}rPr');clear(p)
    for label,f in parts:text_run(p,label,pr);p.append(math(f))
    text_run(p,', averaged over 2D slices/channels.' if i==114 else '.',pr)
    log.append({'paragraph':i,'original':before,'equations':len(parts)})

# Inline occurrences in every section and nested table. Command examples remain literal code.
patterns=[
 (r'alpha=0\.5',eq(atom('α'),n('0.5'))),
 (r'depth=12/154',eq(atom('depth'),frac(n(12),n(154)))),
 (r'normalized gap=0\.5',eq(atom('gap'),n('0.5'))),
 (r'z/\(D-1\)',frac(z,row(atom('D'),op('−'),n(1)))),
 (r'gap/16',frac(atom('gap'),n(16))),
 (r'vmax=vmin\+1',eq(vmax,row(vmin,op('+'),n(1)))),
 (r'reconstruction seed=42\+100\*volume_id\+stride',eq(sub(atom('seed'),'reconstruction'),row(n(42),op('+'),n(100),op('·'),sub(atom('volume'),'id'),op('+'),atom('stride')))),
 (r'Global seed=42',eq(sub(atom('seed'),'global'),n(42))),
 (r'(?:epsilon=1e-7)',eq(eps,power10('−7'))),
 (r'<= 1e-7',row(op('≤'),power10('−7'))),
 (r'sigma(?:=| )([\d.]+)',None),
 (r'eta=0',eq(atom('η'),n(0))),
 (r'(?:D=155|C=1|H=W=240|z=12|z=8|z=16|decay=0\.999|weight decay=1e-4|clip norm=1|scale=0\.18215)',None),
 (r'\b(?:17x256|8x30x30|4x30x30|240x240|30x30|11x11)\b',None),
 (r'\b\d+(?:\.\d+)? \+/- \d+(?:\.\d+)?',None),
 (r'\b2e-5\b',row(n(2),op('×'),power10('−5'))),
 (r'\[-1,1\]',row(op('['),n('−1'),op(','),n(1),op(']'))),
]
combined=re.compile('|'.join('(?P<P'+str(i)+'>'+pattern+')' for i,(pattern,_) in enumerate(patterns)))
for i,p in enumerate(paragraphs):
    if i in formulas:continue
    txt=''.join(p.xpath('.//w:t/text()'))
    if 'Example commands:' in txt:continue
    matches=list(combined.finditer(txt))
    if not matches:continue
    # Only plain text paragraphs; drawing, hyperlink and field XML must remain intact.
    if p.xpath('.//w:drawing | .//w:fldChar | .//w:hyperlink'):continue
    pr=p.find('.//{'+W+'}rPr');clear(p);pos=0
    for m in matches:
        text_run(p,txt[pos:m.start()],pr);s=m.group();j=int(m.lastgroup[1:]);f=patterns[j][1]
        if f is None:
            if 'x' in s and '=' not in s:f=row(*sum(([n(x),op('×')] for x in s.split('x')),[])[:-1])
            elif '+/-' in s:
                a,b=s.split(' +/- ');f=row(n(a),op('±'),n(b))
            elif s.startswith('sigma'):f=eq(atom('σ'),n(re.search(r'[\d.]+',s).group()))
            else:
                a,b=s.split('=',1)
                if a=='H':f=row(atom('H'),op('='),atom('W'),op('='),n(240))
                elif b=='1e-4':f=eq(atom(a),power10('−4'))
                else:f=eq(atom(a),n(b))
        p.append(math(f));pos=m.end()
    text_run(p,txt[pos:],pr)
    log.append({'paragraph':i,'original':txt,'equations':len(matches)})
doc.save(FILE)
def media(path):
    with zipfile.ZipFile(path) as z:return {name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist() if name.startswith('word/media/')}
assert media(FILE)==media(BACKUP)
assert len(doc.tables[0].rows)==67
root=doc._element
counts={k:len(root.findall('.//{'+M+'}'+k)) for k in ['oMath','f','rad','sSub','sSup','acc']}
assert counts['oMath']>50 and counts['f']>=8 and counts['rad']>=5
audit={'docx_sha256':hashlib.sha256(FILE.read_bytes()).hexdigest(),'math_counts':counts,'changes':log,'diagrams_unchanged':True,'outer_rows':67,'authoring':'Microsoft MML2OMML.XSL; editable native OMML'}
(OUT/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='changes'},indent=2))
