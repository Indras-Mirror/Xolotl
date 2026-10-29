#!/usr/bin/env python3
"""Create the reference 640x360 image and 12-second, three-stage shape video.

Needs Pillow, DejaVuSans.ttf and ffmpeg with libx264. Encoder/font versions may
change compressed bytes; the receipt saves versions and input/output hashes.
Use the same generated artifacts on both sides of every numeric or speed test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path,help='new output directory')
    p.add_argument('--font',default='DejaVuSans.ttf')
    p.add_argument('--ffmpeg',default='ffmpeg')
    a=p.parse_args()
    from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
    a.out.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype(a.font,40)
    for index in range(3):
        image=Image.new('RGB',(640,360),'white');draw=ImageDraw.Draw(image)
        if index==0:draw.ellipse((50,110,190,250),fill='red')
        elif index==1:draw.rectangle((250,110,390,250),fill='green')
        else:draw.polygon([(520,100),(440,250),(600,250)],fill='blue')
        draw.text((20,20),str(index+1),font=font,fill='black')
        image.save(a.out/f'{index}.png')
    (a.out/'concat.txt').write_text(''.join(f"file '{i}.png'\nduration 4\n" for i in range(3))+"file '2.png'\n")
    argv=[a.ffmpeg,'-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(a.out/'concat.txt'),
        '-t','12','-r','12','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(a.out/'shapes-12s.mp4')]
    subprocess.run(argv,check=True,timeout=120)
    receipt={'format':'m31-reference-media-v1','pillow':pillow_version,
        'ffmpeg':subprocess.check_output([a.ffmpeg,'-version'],text=True).splitlines()[0],
        'font_sha256':hashlib.sha256(Path(font.path).read_bytes()).hexdigest(),
        'files':{name:hashlib.sha256((a.out/name).read_bytes()).hexdigest() for name in ('0.png','1.png','2.png','shapes-12s.mp4')},
        'geometry':{'width':640,'height':360,'duration_seconds':12,'encoded_fps':12,'sample_fps':1,
                    'stages':['red circle at left','green square at center','blue triangle at right']},'argv':argv}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
