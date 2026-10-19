"""Fetch a project-local renderer from official-listed mirrors; verify SHA-256."""
import concurrent.futures
import hashlib
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.review_tools/libreoffice/renderer_current.msi'
SUFFIX='/tdf/libreoffice/stable/26.2.6/win/x86_64/LibreOffice_26.2.6_Win_x86-64.msi'
URLS=[base+SUFFIX for base in ['https://mirrors.in.sahilister.net','https://mirror.maa.albony.in','https://mirror.kumi.systems','https://mirror.cyberbits.asia']]
EXPECTED='f9877032fd908beb9c0ddf06df4af5c2e85f419c42e14876c4cce5aae5fb2660'

def probe(url):
    start=time.monotonic()
    try:
        with requests.get(url,headers={'Range':'bytes=0-1048575'},stream=True,timeout=(5,8)) as r:
            r.raise_for_status()
            n=0
            for chunk in r.iter_content(262144):
                n+=len(chunk)
                if n>=1048576:break
        return time.monotonic()-start,url
    except requests.RequestException:return 999,url

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    choices=sorted(pool.map(probe,URLS))
print('Mirror selected:',choices[0],flush=True)
with requests.get(choices[0][1],stream=True,timeout=(10,30)) as response:
    response.raise_for_status()
    digest=hashlib.sha256();n=0;next_report=50*1024**2
    with DEST.open('wb') as f:
        for chunk in response.iter_content(1024**2):
            f.write(chunk);digest.update(chunk);n+=len(chunk)
            if n>=next_report:
                print('Downloaded MiB:',n//1024**2,flush=True);next_report+=50*1024**2
assert digest.hexdigest()==EXPECTED,'Renderer checksum does not match official mirror metadata'
print('Verified official renderer checksum; bytes:',n,flush=True)
