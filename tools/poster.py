"""KYOTO NOTES 表紙用：風景写真を、指定パレットだけのフラットな色面（ベクター）に置き換える。
空は写真のグラデーションを使わず、横の色帯（夕焼けの帯）として描き直す。
使い方: python3 poster.py 入力 出力dir"""
import sys, json
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi
TOK={'paper':'#F7F4ED','peach':'#F3C9AB','yellow':'#F0C948','pink':'#F04478','vermilion':'#D64D42','navy':'#273650','ink':'#18191D','paper2':'#EFEADF'}
ORDER=['paper','peach','yellow','paper2','navy','pink','vermilion','ink']   # 下の層から
W=1440                                       # 作業幅
def K(r): r=max(1,int(round(r))); return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def close(m,r): return cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,K(r)).astype(bool)
def opn(m,r): return cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,K(r)).astype(bool)
def largest(m):
    lab,n=ndi.label(m)
    if n==0: return m
    sz=ndi.sum(m,lab,range(1,n+1)); return lab==(1+int(np.argmax(sz)))

def classify(src):
    im=Image.open(src).convert('RGB'); w0,h0=im.size; H=round(h0*W/w0)
    rgb=np.asarray(im.resize((W,H),Image.LANCZOS))
    a=cv2.pyrMeanShiftFiltering(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),9,28)   # 細部を色面へ
    a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB).astype(np.float32)
    R,G,B=a[...,0],a[...,1],a[...,2]; v=a.max(-1); mn=a.min(-1)
    yy=np.arange(H)[:,None]*np.ones((1,W)); xx=np.ones((H,1))*np.arange(W)[None,:]
    sx=W/1920
    # 空：上端につながる、明るくなめらかな領域
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    tex=cv2.GaussianBlur(np.abs(cv2.Laplacian(g,cv2.CV_32F)),(0,0),3)
    skyc=(v>=150)&(tex<6)&(B>=R*0.5)
    lab,n=ndi.label(skyc); top=set(np.unique(lab[:3]))-{0}
    sky=np.isin(lab,list(top)); sky=opn(close(sky,6),4)
    sky=ndi.binary_fill_holes(sky|(yy<5))
    L=np.full((H,W),255,np.uint8); I=lambda n: ORDER.index(n)
    # 夕焼けの帯（空の中の高さで3段）
    L[sky&(yy<330*sx)]=I('paper'); L[sky&(yy>=330*sx)&(yy<520*sx)]=I('peach'); L[sky&(yy>=520*sx)]=I('yellow')
    rest=~sky
    red=(R>G+55)&(R>B+25)
    blue=(B>R+70)&(B>G+40)
    mtn=rest&blue&(R>=45)&(v>=150)
    roof=rest&blue&(R<45)
    L[rest]=I('ink')
    L[mtn]=I('navy')
    L[roof]=I('ink')                             # 屋根はインク。背後の山（紺）と分ける
    L[rest&red&(v>=60)]=I('vermilion')
    L[rest&red&(v>=205)&(G<110)]=I('pink')
    # 街：山と紅葉のあいだは紺の一色にまとめ、明かりだけを黄色の点で残す（細かい建物は描かない）
    city=rest&~red&(((yy>=640*sx)&~mtn)|(yy>=820*sx))&(yy<1080*sx)&(xx<1250*sx)
    L[city]=I('navy')
    # 石垣・建物の明るい面
    stone=rest&~red&~blue&(v>=125)&(xx>=1300*sx)&(yy>=860*sx)
    L[stone]=I('paper2')
    wood=rest&~red&~blue&(R>=150)&(G>=95)&(yy<950*sx)&(xx>=1500*sx)
    L[wood]=I('peach')
    # 街の明かり（小さな黄色い点）
    lights=rest&~red&(v>=195)&(R>=G)&(yy>=780*sx)&(yy<1060*sx)&(xx<1250*sx)
    L[lights]=I('yellow')
    return L,(W,H)

def smooth(L,sig=2.6,min_area=None):
    min_area=min_area or {'yellow':10,'pink':120,'peach':200,'paper2':500,'navy':400,'vermilion':500,'ink':450,'paper':400}
    H,W=L.shape; best=np.full((H,W),-1.0,np.float32); out=np.zeros((H,W),np.uint8)
    for k in range(len(ORDER)):
        p=cv2.GaussianBlur((L==k).astype(np.float32),(0,0),sig)
        m=p>best; best[m]=p[m]; out[m]=k
    for k in range(len(ORDER)):
        m=out==k; lab,n=ndi.label(m)
        if n==0: continue
        sz=ndi.sum(m,lab,range(1,n+1)); small=np.zeros(n+1,bool); small[1:]=sz<min_area.get(ORDER[k],300)
        kill=small[lab]
        if kill.any():
            idx=ndi.distance_transform_edt(kill,return_distances=False,return_indices=True)
            out[kill]=out[idx[0][kill],idx[1][kill]]
    return out

def fmt(p,sc): return f"{round(p[0]*sc,1):g},{round(p[1]*sc,1):g}"
def contour_path(c,eps,sc):
    c=cv2.approxPolyDP(c,eps,True)[:,0,:].astype(np.float32); n=len(c)
    if n<3: return ''
    prv=np.roll(c,1,0); nxt=np.roll(c,-1,0); v1=c-prv; v2=nxt-c
    cos=(v1*v2).sum(1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)+1e-9)
    corner=cos<0.34; mid=(c+nxt)/2
    d=[f"M{fmt(mid[-1],sc)}"]
    for i in range(n): d.append(f"L{fmt(c[i],sc)}L{fmt(mid[i],sc)}" if corner[i] else f"Q{fmt(c[i],sc)} {fmt(mid[i],sc)}")
    return ''.join(d)+'Z'
def mask_path(m,eps,sc,min_area=6):
    cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    return ''.join(contour_path(c,eps,sc) for c in cs if cv2.contourArea(c)>=min_area)

def build(src,outdir,vw=1536):
    L,(w,h)=classify(src); L=smooth(L)
    sc=vw/w; vh=round(h*sc)
    layers={}
    for i,name in enumerate(ORDER):
        if i==0: continue
        m=np.zeros_like(L,bool)
        for j in range(i,len(ORDER)):
            pass
        m=(L==i)
        # 下の層に重ねて描くので、各色は自分の領域だけ（少し膨らませて継ぎ目を消す）
        m=cv2.dilate(m.astype(np.uint8),K(1)).astype(bool)
        layers[name]=mask_path(m,1.4,sc)
    json.dump({'w':vw,'h':vh,'base':'paper','order':ORDER,'layers':layers},open(f'{outdir}/scene.json','w'))
    pal=np.array([[int(TOK[n][i:i+2],16) for i in (1,3,5)] for n in ORDER],np.uint8)
    Image.fromarray(pal[L]).save(f'{outdir}/scene-preview.png')
    print('bytes',{k:len(v) for k,v in layers.items()},sum(len(v) for v in layers.values()))
if __name__=='__main__': build(sys.argv[1],sys.argv[2])
