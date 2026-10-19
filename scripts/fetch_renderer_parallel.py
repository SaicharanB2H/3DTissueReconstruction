"""Download official LibreOffice MSI in resumable ranges and verify its digest."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, time, hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.review_tools/libreoffice'
SUFFIX='/tdf/libreoffice/stable/26.2.6/win/x86_64/LibreOffice_26.2.6_Win_x86-64.msi'
BASES=['https://mirrors.in.sahilister.net','https://mirror.maa.albony.in','https://mirror.kumi.systems','https://mirror.cyberbits.asia','https://mirror.twds.com.tw','https://mirrors.nju.edu.cn','https://mirror-hk.koddos.net']
SIZE=373252096
EXPECTED='f9877032fd908beb9c0ddf06df4af5c2e85f419c42e14876c4cce5aae5fb2660'
CHUNK=4*1024*1024
PARTS=OUT/'parts'; PARTS.mkdir(parents=True,exist_ok=True)
def probe(base):
    start=time.monotonic()
    try:
        r=requests.get(base+SUFFIX,headers={'Range':'bytes=0-262143'},timeout=(5,10))
        if r.status_code!=206 or len(r.content)!=262144:return None
        return time.monotonic()-start,base
    except requests.RequestException:return None
with ThreadPoolExecutor(max_workers=7) as pool:
    mirrors=sorted(x for x in pool.map(probe,BASES) if x)
print('Range-capable mirrors:',mirrors,flush=True)
assert mirrors,'No range-capable mirror'
def getpart(i):
    start=i*CHUNK;end=min(SIZE,start+CHUNK)-1
    p=PARTS/f'{i:03d}.part'
    if p.exists() and p.stat().st_size==end-start+1:return i
    for attempt in range(8):
        base=mirrors[(i+attempt)%len(mirrors)][1]
        try:
            r=requests.get(base+SUFFIX,headers={'Range':f'bytes={start}-{end}'},timeout=(10,45))
            if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}' or len(r.content)!=end-start+1:
                raise ValueError('Invalid byte range response')
            p.write_bytes(r.content);return i
        except (requests.RequestException,ValueError) as e:
            if attempt==7:raise RuntimeError(f'Part {i} failed: {e}')
count=(SIZE+CHUNK-1)//CHUNK
with ThreadPoolExecutor(max_workers=16) as pool:
    for n,f in enumerate(as_completed([pool.submit(getpart,i) for i in range(count)]),1):
        f.result()
        if n%10==0 or n==count:print(f'Completed {n}/{count} ranges',flush=True)
target=OUT/'renderer_verified.msi'
digest=hashlib.sha256()
with target.open('wb') as f:
    for i in range(count):
        data=(PARTS/f'{i:03d}.part').read_bytes();f.write(data);digest.update(data)
assert digest.hexdigest()==EXPECTED,'Official SHA-256 mismatch'
print('Verified MSI SHA-256:',digest.hexdigest(),flush=True)
