"""Create a plain, numbered architecture drawing from the implemented pipeline."""
from pathlib import Path
from html import escape
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'patent_architecture'
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1500, 2180
SCALE = 2
im = Image.new('RGB', (W * SCALE, H * SCALE), 'white')
draw = ImageDraw.Draw(im)
font_path = 'C:/Windows/Fonts/arial.ttf'
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<rect width="100%" height="100%" fill="white"/>']

def text(x, y, value, size=25):
    font = ImageFont.truetype(font_path, size * SCALE)
    draw.text((x * SCALE, y * SCALE), value, font=font, fill='black', anchor='mm')
    svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" font-family="Arial,Helvetica,sans-serif" font-size="{size}" fill="black">{escape(value)}</text>')

def line(points, arrow=True):
    draw.line([(int(x*SCALE), int(y*SCALE)) for x,y in points], fill='black', width=3*SCALE)
    svg.append('<polyline points="' + ' '.join(f'{x},{y}' for x,y in points) + '" fill="none" stroke="black" stroke-width="3"/>')
    if arrow:
        x,y=points[-1]; px,py=points[-2]
        a=math.atan2(y-py,x-px)
        pts=[(x,y),(x-14*math.cos(a-.44),y-14*math.sin(a-.44)),(x-14*math.cos(a+.44),y-14*math.sin(a+.44))]
        draw.polygon([(int(a*SCALE),int(b*SCALE)) for a,b in pts], fill='black')
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in pts)+'" fill="black"/>')

def box(x,y,w,h,number,labels,size=25):
    draw.rectangle((x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE), outline='black', width=3*SCALE)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="black" stroke-width="3"/>')
    lines=[str(number)]+labels
    step=34
    for i,label in enumerate(lines):
        text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*step,label,size)

# Main flow, split conditioning branches and sampling inputs.
box(500,70,500,110,100,['Sparse MRI slices','and acquired slice indices'])
box(500,240,500,120,110,['Intensity normalization','using acquired slices'])
box(500,420,500,120,120,['Bounding-slice selection','and target-depth calculation'])
box(240,620,440,130,130,['Shared residual','slice encoder'])
box(820,620,440,130,140,['Fourier depth, offset','and gap embedding'])
box(500,840,500,130,150,['Conditioning formation','Context tokens and spatial maps'])
box(500,1070,500,130,160,['Conditional latent U-Net','Noise prediction'])
box(70,1070,330,130,170,['Seeded Gaussian','latent noise'])
box(1100,1070,330,130,180,['DDIM sampler','and timestep schedule'])
box(500,1330,500,110,190,['VAE decoder','Missing-slice synthesis'])
box(500,1530,500,110,200,['Volume assembly','Insert generated slices'])
box(500,1720,500,110,210,['Optional depth smoothing'])
box(500,1910,500,110,220,['Restore acquired slices','Dense MRI volume output'])

line([(750,180),(750,240)])
line([(750,360),(750,420)])
line([(620,540),(620,580),(460,580),(460,620)])
line([(880,540),(880,580),(1040,580),(1040,620)])
line([(460,750),(460,800),(620,800),(620,840)])
line([(1040,750),(1040,800),(880,800),(880,840)])
line([(750,970),(750,1070)])
text(855,1015,'Conditioning',23)
line([(400,1135),(500,1135)])
line([(1000,1110),(1100,1110)])
line([(1100,1160),(1000,1160)])
text(1050,1082,'Noise',19)
text(1050,1190,'Latent, t',19)
# The sampler holds the updated latent and passes the final latent to decoding.
line([(1265,1200),(1265,1270),(750,1270),(750,1330)])
text(990,1240,'Final denoised latent',23)
line([(750,1440),(750,1530)])
line([(750,1640),(750,1720)])
line([(750,1830),(750,1910)])
# Original normalized measurements bypass generation and smoothing.
line([(500,300),(30,300),(30,1965),(500,1965)])
text(250,260,'Normalized acquired slices',23)
line([(30,1585),(500,1585)])

text(750,2090,'FIG. 1',30)
text(750,2135,'SPARSE-TO-DENSE MRI RECONSTRUCTION ARCHITECTURE',24)
svg.append('</svg>')
(OUT / 'patent_architecture.svg').write_text('\n'.join(svg),encoding='utf-8')
im.save(OUT / 'patent_architecture.png', dpi=(300,300))
print(OUT / 'patent_architecture.svg')
print(OUT / 'patent_architecture.png')
