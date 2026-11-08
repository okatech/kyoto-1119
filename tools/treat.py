"""KYOTO NOTES の画像加工：指定パレットへの減色＋色面の整理（グラデーション除去）。
使い方: python3 treat.py 入力 出力.png [幅]"""
import sys
from PIL import Image, ImageFilter
import numpy as np
PAL = {  # 指示書のトークン＋影用に1段だけ派生色
 'ink':(0x18,0x19,0x1D),'navy':(0x27,0x36,0x50),'paper':(0xF7,0xF4,0xED),
 'peach':(0xF3,0xC9,0xAB),'peach_shade':(0xD9,0xA4,0x86),'yellow':(0xF0,0xC9,0x48),
}
def to_lab(a):
    a=a/255.0; a=np.where(a>0.04045,((a+0.055)/1.055)**2.4,a/12.92)
    M=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])
    xyz=a@M.T/np.array([0.9505,1.0,1.089])
    f=np.where(xyz>0.008856,np.cbrt(xyz),7.787*xyz+16/116)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)
def treat(src,dst,width=1400):
    im=Image.open(src).convert('RGB'); im.thumbnail((width,width))
    im=im.filter(ImageFilter.MedianFilter(5))            # 細かい毛束・ノイズを色面へ
    a=np.asarray(im).astype(float)
    names=list(PAL); P=np.array([PAL[n] for n in names],float)
    d=((to_lab(a)[...,None,:]-to_lab(P)[None,None])**2).sum(-1)
    idx=d.argmin(-1)
    # 背景のにじみ（暖色の暗い光）を消し、明るい毛先だけを黄色の色面にする
    r,g,b=[a[...,i]/255 for i in range(3)]
    mx=np.max(a,-1)/255; mn=np.min(a,-1)/255; sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1e-6),0)
    hue=np.degrees(np.arctan2(np.sqrt(3)*(g-b),2*r-g-b))%360
    warm=(hue>=28)&(hue<=55)&(sat>=0.30)
    idx=np.where(warm&(mx<0.76),names.index('ink'),idx)
    idx=np.where(warm&(mx>=0.76)&(sat>=0.40),names.index('yellow'),idx)
    out=Image.fromarray(P[idx].astype('uint8'))
    out=out.filter(ImageFilter.ModeFilter(7))             # 境界の粒を消して面を整える
    if dst.lower().endswith('.png'):                      # 色数が少ないのでPNGが軽くて正確
        pal=Image.new('P',(1,1)); flat=[int(c) for n in names for c in PAL[n]]
        pal.putpalette(flat+[0]*(768-len(flat)))           # 指定パレットに厳密に固定
        out.quantize(palette=pal,dither=Image.Dither.NONE).save(dst,optimize=True)
    else:
        out.save(dst,quality=86,optimize=True,progressive=True,subsampling=0)
if __name__=='__main__':
    treat(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 1400)
