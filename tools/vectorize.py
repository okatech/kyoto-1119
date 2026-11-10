"""KYOTO NOTES 表紙イラストのベクター化。
フラットな色面（指定パレット）に整理し、髪は一つのシルエット、輪郭は均一なインクの線にして SVG で出力する。
使い方: python3 vectorize.py 入力画像 出力dir"""
import sys, json
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi

S=2                                  # 作業倍率（輪郭をなめらかにするため2倍で処理）
TOK={'ink':'#18191D','navy':'#273650','peach':'#F3C9AB','shade':'#D9A486','yellow':'#F0C948','paper':'#F7F4ED'}
ORDER=['ink','navy','peach','shade','yellow','paper']

def disk(r):
    r=int(round(r)); y,x=np.ogrid[-r:r+1,-r:r+1]; return (x*x+y*y<=r*r)
def K(r):
    r=max(1,int(round(r))); return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def close(m,r): return cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,K(r)).astype(bool)   # 画像の端では削らない
def opn(m,r): return cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,K(r)).astype(bool)
def dil(m,r): return cv2.dilate(m.astype(np.uint8),K(r)).astype(bool)
def soft(m,sig): return cv2.GaussianBlur(m.astype(np.float32),(0,0),sig)>0.5
def largest(m):
    lab,n=ndi.label(m)
    if n==0: return m
    sz=ndi.sum(m,lab,range(1,n+1)); return lab==(1+int(np.argmax(sz)))
def drop_small(m,area):
    lab,n=ndi.label(m)
    if n==0: return m
    sz=ndi.sum(m,lab,range(1,n+1)); keep=np.zeros(n+1,bool); keep[1:]=sz>=area; return keep[lab]

def segment(src):
    im=Image.open(src).convert('RGB'); W0,H0=im.size
    a=np.asarray(im.resize((W0*S,H0*S),Image.LANCZOS)).astype(np.float32)
    a=cv2.GaussianBlur(a,(0,0),0.8)
    H,W=a.shape[:2]; R,G,B=a[...,0],a[...,1],a[...,2]
    v=a.max(-1); br=B/np.maximum(R,1)
    # --- 明るい面（肌・影・金の毛先・白） ---
    bright=v>=150
    white=(v>=225)&((v-a.min(-1))<=28)
    ycore=(R>=228)&(G>=0.76*R)&(br<0.60)             # 金の毛束の芯（背景のにじみは暗いので入らない）
    cand=bright&~white&~ycore
    main=largest(opn(cand&(br>=0.585),1.5*S))         # 顔〜首のひとつながり
    yz=dil(drop_small(close(ycore,4*S),400*S*S),8*S)  # 毛先のまわり
    skin=main|(cand&(v>=200)&~yz)                     # レンズ越し・前髪の間の肌も肌のまま
    frag=cand&yz&~skin&(R>=205)                       # 毛先まわりの明るい端切れは金に寄せる
    yellow=close(ycore|frag,6*S)&~skin
    yellow=drop_small(opn(yellow,2.5*S),500*S*S)
    yellow=soft(ndi.binary_fill_holes(yellow),4*S)&~skin   # 毛束の集まりを、ひとつの面に
    # --- 上着・シャツ（青みのある中間の暗さ） ---
    cloth=(v>=48)&(v<=112)&(B>=R+9)
    cloth[:int(H*0.50)]=False
    cloth=opn(cloth,2.5*S)
    # --- 髪：細い毛束の線が密にある範囲を一つのシルエットにまとめる ---
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9*S,9*S))
    top=v-cv2.morphologyEx(v,cv2.MORPH_OPEN,k)
    strand=(top>=7)&(v<120)&(B>=R); strand[int(H*0.62):]=False
    strand=drop_small(strand,30*S*S)
    hs=ndi.binary_fill_holes(close(strand,10*S))
    vb=cv2.GaussianBlur(v,(0,0),1.5*S)
    core=(vb>=15.3)&(vb<=34)&(br>=0.95)&(br<=1.7)
    hair=hs|(core&dil(hs,5*S))
    hair=opn(close(hair,4*S),5*S)
    # 髪と金の毛先のあいだ（暗い髪が続く部分）をつなぐ
    hair=hair|(close(hair|yellow,7*S)&core)
    hair=largest(opn(soft(hair,3*S),7*S)|skin)&~skin&~yellow   # はねた毛は落とし、輪郭を一筆のシルエットに
    # --- 人物の外形 ---
    F=hair|skin|yellow|white|cloth
    F=close(F,6*S)
    low=np.zeros_like(F); low[int(H*0.66):]=True
    F=F|(close(F&low,20*S)&low)                       # 服の輪郭線・しわ・襟の影（暗い形）で外とつながる隙間をふさぐ
    pad=np.pad(F,((0,2),(2,0)),constant_values=False); pad[-2:,:]=True; pad[int(H*0.62):,:2]=True
    F=ndi.binary_fill_holes(pad)[:-2,2:]
    F=largest(opn(F,2*S))
    F=soft(F,2.0*S)
    # --- 色面の割り当て（人物の内側だけ） ---
    L=np.zeros((H,W),np.uint8)                        # 0=ink（線・髪・影の形）
    L[cloth]=1
    shade=skin&((R<241)|(G<190))
    L[skin&~shade]=2; L[shade]=3; L[yellow]=4; L[white&F]=5
    L[hair]=0
    L[~F]=255
    return L,F,(W0,H0)

def smooth_labels(L,F):
    """境界のギザギザを取る：クラスごとにぼかして多数決。線（インク）は消えないよう少し優遇。"""
    H,W=L.shape; best=np.full((H,W),-1.0,np.float32); out=np.zeros((H,W),np.uint8)
    for k in range(len(ORDER)):
        p=cv2.GaussianBlur((L==k).astype(np.float32),(0,0),1.1*S)
        if k==0: p*=1.18
        m=p>best; best[m]=p[m]; out[m]=k
    # 小さな粒は周りの色に吸収
    for k in range(len(ORDER)):
        m=(out==k)&F; lab,n=ndi.label(m)
        if n==0: continue
        sz=ndi.sum(m,lab,range(1,n+1)); small=np.zeros(n+1,bool); small[1:]=sz<(14*S*S if k==0 else 40*S*S)
        kill=small[lab]
        if kill.any():
            idx=ndi.distance_transform_edt(kill|~F|(out==k)&False,return_distances=False,return_indices=True)
            valid=~kill&F
            idx=ndi.distance_transform_edt(~valid,return_distances=False,return_indices=True)
            out[kill]=out[idx[0][kill],idx[1][kill]]
    out[~F]=255
    return out

def fmt(p): return f"{p[0]/S:.1f},{p[1]/S:.1f}".replace('.0,',',').rstrip('0').rstrip('.') if False else f"{round(p[0]/S,1):g},{round(p[1]/S,1):g}"
def contour_path(c,eps):
    c=cv2.approxPolyDP(c,eps,True)[:,0,:].astype(np.float32)
    n=len(c)
    if n<3: return ''
    prv=np.roll(c,1,0); nxt=np.roll(c,-1,0)
    v1=c-prv; v2=nxt-c
    cosang=(v1*v2).sum(1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)+1e-9)
    corner=cosang<0.34                                # 70度より鋭い折れは角として残す
    mid=(c+nxt)/2
    d=[f"M{fmt(mid[-1])}"]
    for i in range(n):
        d.append(f"L{fmt(c[i])}L{fmt(mid[i])}" if corner[i] else f"Q{fmt(c[i])} {fmt(mid[i])}")
    return ''.join(d)+'Z'
def mask_path(m,eps=2.0,min_area=10):
    cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    return ''.join(contour_path(c,eps) for c in cs if cv2.contourArea(c)>=min_area*S*S)

def build(src,outdir):
    L,F,(W0,H0)=segment(src)
    L=smooth_labels(L,F)
    outline=dil(F,3*S)
    layers={"outline":mask_path(outline,eps=2.6)}
    for i,name in enumerate(ORDER[1:],start=1):
        layers[name]=mask_path((L>=i)&(L!=255))
    json.dump({'w':W0,'h':H0,'layers':layers},open(f'{outdir}/cover-layers.json','w'))
    # 確認用のラスター
    pal=np.array([[int(TOK[n][i:i+2],16) for i in (1,3,5)] for n in ORDER],np.uint8)
    prev=np.full(L.shape+(3,),(240,68,120),np.uint8); prev[outline]=pal[0]
    ok=L!=255; prev[ok]=pal[L[ok]]
    Image.fromarray(prev).resize((W0,H0),Image.LANCZOS).save(f'{outdir}/cover-preview.png')
    print({k:len(v) for k,v in layers.items()}, 'total', sum(len(v) for v in layers.values()))
if __name__=='__main__': build(sys.argv[1],sys.argv[2])
