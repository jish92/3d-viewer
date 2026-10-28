import sys,os; sys.path.insert(0,'/workspace/me3d/tshirt_shorts')
import numpy as np, trimesh, scipy.sparse as sp
from scipy.spatial import cKDTree
from clean_body import adjacency, refine, smoothstep
SRC='/workspace/me3d/tshirt_shorts/jackson_tshirt_shorts_v4_clean.glb'
VARIANT=sys.argv[1]; OUT=sys.argv[2]; STAGE=sys.argv[3] if len(sys.argv)>3 else 'full'
s=trimesh.load(SRC); head=s.geometry['geometry_0']; body=s.geometry['geometry_1']
V=body.vertices.astype(float).copy(); F=body.faces.copy(); C=np.array(body.visual.vertex_colors)[:,:3].astype(float)
N0=len(V)
# skirt verts = last 2*nP verts (duplicated coincident positions at the neck): never move anything near them
Lm=np.linalg.norm(V,axis=1)
skirtV=V[(V[:,1]>1.5)]  # fallback; refined below
# find skirt: verts coincident with head boundary
th=cKDTree(head.vertices); dh,_=th.query(V); pinned_core=dh<1e-5
print('pinned (head-coincident) verts',pinned_core.sum())
# freeze zone = near head-coincident verts or any vertex above the neck line
fz=cKDTree(V[pinned_core]); dfz,_=fz.query(V)
def move_mult(P=None):
    d=dfz
    return smoothstep(d,0.015,0.05)*(V[:,1]<1.56)
r=C[:,0]/np.maximum(C.sum(1),1)
skin0=smoothstep(r,0.47,0.64)           # 1 = skin colour
cloth=1-skin0
A=adjacency(F,len(V)); deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
def avg(x): return (A@x)/(deg[:,None] if x.ndim>1 else deg)
def spread(x,it):
    for _ in range(it): x=np.maximum(x,avg(x))
    return x
def sdist(P,a,b):
    ab=b-a; t=np.clip(((P-a)@ab)/(ab@ab),0,1); return np.linalg.norm(P-(a+t[:,None]*ab),axis=1),t
def vnormals(V,F):
    m=trimesh.Trimesh(V,F,process=False); return m.vertex_normals.copy()
def G(x,c,sg): return np.exp(-0.5*((x-c)/sg)**2)
# ---- armature-ish geometry
SH=np.array([0.21,1.37,-0.065]); EL=np.array([0.40,1.12,-0.035]); WR=np.array([0.56,0.90,0.025])
def arm_fields(V):
    out=[]
    for sx in (1,-1):
        m=np.array([sx,1,1])
        d1,t1=sdist(V,SH*m,EL*m); d2,t2=sdist(V,EL*m,WR*m); out.append((d1,t1,d2,t2))
    return out
af=arm_fields(V)
darm=np.minimum.reduce([a[0] for a in af]+[a[2] for a in af])
armness=1-smoothstep(darm,0.075,0.115)           # 1 in arm tube, 0 in torso
# ---------- 1. wrinkle removal (weighted Taubin) on cloth ----------
mm=move_mult()
def taubin(V,w,its,lam=0.5,mu=-0.53):
    V=V.copy()
    for _ in range(its):
        for k in (lam,mu):
            d=avg(V)-V; V+=k*w[:,None]*d
    return V
wc=np.clip(spread(cloth.copy(),6)*1.0,0,1)*mm      # cloth plus a 3-ring skirt around it
# shoes/foot smoothing weight handled in section 5
V1=taubin(V,wc,45)
# ---------- 2. torso reshaping ----------
zc=-0.04
y=V1[:,1]
def prof(y,pts): return np.interp(y,[p[0] for p in pts],[p[1] for p in pts])
sx=prof(y,[(0.80,1.0),(0.90,0.93),(1.00,0.88),(1.10,0.855),(1.20,0.87),(1.30,0.95),(1.38,1.0),(1.60,1.0)])
szf=prof(y,[(0.80,1.0),(0.88,0.93),(1.00,0.80),(1.12,0.80),(1.24,0.87),(1.34,0.97),(1.42,1.0),(1.60,1.0)])  # front depth factor
szb=prof(y,[(0.80,1.0),(0.88,0.95),(1.00,0.86),(1.12,0.86),(1.24,0.90),(1.34,0.97),(1.42,1.0),(1.60,1.0)])  # back
wT=(1-armness)*mm*(y>0.80)
V2=V1.copy()
z=V1[:,2]-zc; fac=np.where(z>0,szf,szb)
V2[:,0]=V1[:,0]+wT*(V1[:,0]*sx-V1[:,0]); V2[:,2]=V1[:,2]+wT*(zc+z*fac-V1[:,2])
# ---------- 3. arms: shrink the sleeve tubes ----------
Vn=V2.copy()
for sg,(d1,t1,d2,t2) in zip((1,-1),af):
    m=np.array([sg,1,1]); 
    use1=d1<=d2
    d=np.where(use1,d1,d2); 
    # closest axis point
    ab1=(EL-SH)*m; p1=SH*m+np.clip(((V2-SH*m)@ab1)/(ab1@ab1),0,1)[:,None]*ab1
    ab2=(WR-EL)*m; p2=EL*m+np.clip(((V2-EL*m)@ab2)/(ab2@ab2),0,1)[:,None]*ab2
    ax=np.where(use1[:,None],p1,p2)
    side=(V2[:,0]*sg>0.12)&(d<0.13)
    # sleeve-ness: cloth amount smoothed a little, only on the arm tube
    k=1-0.24*np.clip(spread(cloth.copy(),2),0,1)
    w=side*mm*(1-smoothstep(d,0.10,0.13))
    # fade toward the torso (near shoulder) so we do not tear the armpit/shoulder seam
    wsh=smoothstep(V2[:,0]*sg,0.20,0.27)
    Vn=Vn+(w*wsh*(k-1))[:,None]*(V2-ax)*np.array([0,0,0])[None,:]*0   # placeholder (applied below)
    Vn[:,:]=Vn+((w*wsh*(k-1))[:,None]*(V2-ax))
V3=Vn
# ---------- 4. legs: slim the shorts tubes & remove hem ridge ----------
def leg_axis(sg): return np.array([0.115*sg,0.88,-0.04]),np.array([0.135*sg,0.47,-0.065]),np.array([0.14*sg,0.10,-0.045])
Vl=V3.copy()
for sg in (1,-1):
    h,k,a=leg_axis(sg); dd1,tt1=sdist(V3,h,k); dd2,tt2=sdist(V3,k,a)
    use1=dd1<=dd2; ab1=k-h; p1=h+np.clip(((V3-h)@ab1)/(ab1@ab1),0,1)[:,None]*ab1; ab2=a-k; p2=k+np.clip(((V3-k)@ab2)/(ab2@ab2),0,1)[:,None]*ab2
    ax=np.where(use1[:,None],p1,p2); d=np.where(use1,dd1,dd2)
    sel=((V3[:,0]*sg)>0.0)&(d<0.16)&(V3[:,1]<0.88)&(V3[:,1]>0.30)
    kk=1-0.08*np.clip(spread(cloth.copy(),2),0,1)
    w=sel*mm
    Vl=Vl+((w*(kk-1))[:,None]*(V3-ax))*np.array([1,0,1])
V4=Vl
def profile_fix(V,pts,sel_fn,urange,sig,dmax,blend=1.0,planar=False,lo=0.80,hi=1.04):
    segs=[(pts[i],pts[i+1]) for i in range(len(pts)-1)]; Ls=[np.linalg.norm(b-a) for a,b in segs]; off=np.cumsum([0]+Ls)
    best=None
    for (a,b),o in zip(segs,off):
        ab=b-a; tt=np.clip(((V-a)@ab)/(ab@ab),0,1); p=a+tt[:,None]*ab; d=np.linalg.norm(V-p,axis=1); u=o+tt*np.linalg.norm(ab)
        if best is None: best=(d,u,p)
        else:
            m=d<best[0]; best=(np.where(m,d,best[0]),np.where(m,u,best[1]),np.where(m[:,None],p,best[2]))
    d,u,p=best
    sel=sel_fn(V)&(d<dmax)&(u>urange[0])&(u<urange[1])
    bins=np.arange(urange[0],urange[1]+0.01,0.01); bi=np.clip(((u-urange[0])/0.01).astype(int),0,len(bins)-1)
    med=np.full(len(bins),np.nan)
    for k in range(len(bins)):
        m=sel&(bi==k)
        if m.sum()>20: med[k]=np.median(d[m])
    ok=~np.isnan(med); med=np.interp(np.arange(len(bins)),np.where(ok)[0],med[ok])
    tg=np.convolve(np.pad(med,15,mode='edge'),np.exp(-0.5*(np.arange(-15,16)*0.01/sig)**2)/np.exp(-0.5*(np.arange(-15,16)*0.01/sig)**2).sum(),mode='valid')
    ratio=np.clip(tg/med,lo,hi)[bi]
    w=sel*mm*blend
    return V+((w*(ratio-1))[:,None]*(V-p))*(np.array([1,0,1]) if planar else 1)
def tube_fix(V,pts,mask_fn,uref,uz,slope,fade,cl,planar=False):
    segs=[(pts[i],pts[i+1]) for i in range(len(pts)-1)]; Ls=[np.linalg.norm(b-a) for a,b in segs]; off=np.cumsum([0]+Ls)
    best=None
    for (a,b),o in zip(segs,off):
        ab=b-a; tt=np.clip(((V-a)@ab)/(ab@ab),0,1); p=a+tt[:,None]*ab; d=np.linalg.norm(V-p,axis=1); u=o+tt*np.linalg.norm(ab); dirv=np.broadcast_to(ab/np.linalg.norm(ab),V.shape)
        if best is None: best=(d,u,p,dirv)
        else:
            m=d<best[0]; best=(np.where(m,d,best[0]),np.where(m,u,best[1]),np.where(m[:,None],p,best[2]),np.where(m[:,None],dirv,best[3]))
    d,u,p,a=best
    e1=np.cross(a,[0,0,1.0]); e1/=np.linalg.norm(e1,axis=1,keepdims=True); e2=np.cross(a,e1)
    r=V-p; th=np.arctan2((r*e2).sum(1),(r*e1).sum(1)); rho=np.linalg.norm(r,axis=1)
    NB=24; bi=((th+np.pi)/(2*np.pi)*NB).astype(int)%NB
    ref=(u>uref[0])&(u<uref[1])&(d<0.2)&mask_fn(V)&(cl<0.3)
    med=np.zeros(NB)
    for k in range(NB):
        m=ref&(bi==k); med[k]=np.median(rho[m]) if m.sum()>3 else np.nan
    ok=~np.isnan(med); med=np.interp(np.arange(NB),np.where(ok)[0],med[ok],period=NB)
    for _ in range(3): med=0.25*np.roll(med,1)+0.5*med+0.25*np.roll(med,-1)
    # smooth interpolation across theta
    fb=(th+np.pi)/(2*np.pi)*NB-0.5; k0=np.floor(fb).astype(int); fr=fb-k0; rr=(1-fr)*med[k0%NB]+fr*med[(k0+1)%NB]
    um=0.5*(uref[0]+uref[1]); rt=rr+slope*(um-u)
    zone=mask_fn(V)&(u>uz[0])&(u<uz[1])&(d<0.2)
    b=1-smoothstep(u,uz[0],uz[0]+fade)
    rn=b*rho+(1-b)*rt
    w=zone*mm
    k=np.where(rho>1e-6,rn/np.maximum(rho,1e-6),1.0)
    return V+((w*(k-1))[:,None]*r)
for sg in (1,-1):
    m_=np.array([sg,1,1])
    V4=tube_fix(V4,[SH*m_,EL*m_,WR*m_],lambda P,sg=sg:(P[:,0]*sg>0.22),(0.14,0.20),(0.05,0.14),0.16,0.035,cloth)
    h,k,a=leg_axis(sg)
    V4=tube_fix(V4,[h,k,a],lambda P,sg=sg:(P[:,0]*sg>0.0)&(P[:,1]<0.9),(0.38,0.44),(0.10,0.38),0.10,0.12,cloth)
# ---------- 5. bare feet: replace the sneakers by a procedural foot (marching cubes) stitched to the leg at y=0.20 ----------
sys.path.insert(0,'/workspace/me3d/swim'); from foot import build_foot
YC=0.20
cenF=V4[F].mean(1); delF=(cenF[:,1]<YC)&(np.abs(cenF[:,0])>0.04)
Fk=F[~delF]
Vn=[V4]; Cn=[C]; clo=[cloth]; Fn=[Fk]; nv=len(V4)
# old boundary vertices at the cut
used=np.zeros(len(V4),bool); used[Fk.ravel()]=True
for sg in (1,-1):
    fv,ff=build_foot(sg)
    # drop the cap faces and keep only the open ankle ring
    cap=(fv[ff][:,:,1]>YC-1e-4).all(1); ff=ff[~cap]
    # boundary loop of the foot: edges used once
    e=np.sort(np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),1); uq,cnt=np.unique(e,axis=0,return_counts=True); be=uq[cnt==1]
    bv=np.unique(be); bv=bv[fv[bv][:,1]>YC-0.01]; 
    # old cut loop: boundary verts of Fk near the cut plane on this side
    e2=np.sort(np.concatenate([Fk[:,[0,1]],Fk[:,[1,2]],Fk[:,[2,0]]]),1); u2,c2=np.unique(e2,axis=0,return_counts=True); b2=np.unique(u2[c2==1])
    ob=b2[(np.abs(V4[b2,1]-YC)<0.02)&(V4[b2,0]*sg>0.04)]
    cxy=np.array([0.1355*sg,-0.066]); 
    def ang(P): return np.arctan2(P[:,2]-cxy[1],P[:,0]-cxy[0])
    a_new=ang(fv[bv]); a_old=ang(V4[ob]); o1=np.argsort(a_new); o2=np.argsort(a_old); bv=bv[o1]; ob=ob[o2]; a_new=a_new[o1]; a_old=a_old[o2]
    base=nv; nv+=len(fv)
    # bridge (zipper merge by angle, cyclic)
    ia=0; ib=0; na=len(bv); nb=len(ob); tris=[]
    # start both at the smallest angle; unwrap old angles beyond new
    A_=list(a_new)+[a_new[0]+2*np.pi]; B_=list(a_old)+[a_old[0]+2*np.pi]
    # align starts
    j0=int(np.argmin(np.abs(np.unwrap(a_old)-a_new[0]))) if False else int(np.argmin(np.abs(((a_old-a_new[0]+np.pi)%(2*np.pi))-np.pi)))
    ob=np.roll(ob,-j0); a_old=np.roll(a_old,-j0); a_old=np.where(a_old<a_new[0]-1e-9,a_old+2*np.pi,a_old); 
    B_=list(a_old)+[a_old[0]+2*np.pi]; ia=ib=0
    while ia<na or ib<nb:
        if ia<na and (ib>=nb or A_[ia+1]<=B_[ib+1]+1e-12 and ia<na):
            tris.append((base+bv[ia],ob[ib%nb],base+bv[(ia+1)%na])); ia+=1
        else:
            tris.append((base+bv[ia%na],ob[ib%nb],ob[(ib+1)%nb])); ib+=1
        if ia>=na and ib>=nb: break
    tris=np.array(tris)
    dff=set(map(tuple,(np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]])+base).tolist()))
    dst=[(int(a),int(b)) for t_ in tris for a,b in ((t_[0],t_[1]),(t_[1],t_[2]),(t_[2],t_[0]))]
    same=sum(1 for e in dst if e in dff and (e[1],e[0]) not in dff); opp=sum(1 for e in dst if (e[1],e[0]) in dff)
    print('bridge tris',len(tris),'edges same-dir as foot',same,'opposite',opp)
    if same>opp: tris=tris[:,::-1]
    dold=set(map(tuple,np.concatenate([Fk[:,[0,1]],Fk[:,[1,2]],Fk[:,[2,0]]]).tolist())); dst=[(int(a),int(b)) for t_ in tris for a,b in ((t_[0],t_[1]),(t_[1],t_[2]),(t_[2],t_[0]))]
    print('  vs old leg: same',sum(1 for e in dst if e in dold and (e[1],e[0]) not in dold),'opp',sum(1 for e in dst if (e[1],e[0]) in dold))
    Fn.append(ff+base); Fn.append(tris); Vn.append(fv)
    # colours: nearest old skin vertex above the cut
    tref=cKDTree(V4[(V4[:,1]>YC)&(V4[:,1]<YC+0.06)&(V4[:,0]*sg>0.04)]); cref=C[(V4[:,1]>YC)&(V4[:,1]<YC+0.06)&(V4[:,0]*sg>0.04)]
    _,jj=tref.query(fv); Cn.append(cref[jj]); clo.append(np.zeros(len(fv)))
V4=np.vstack(Vn); C=np.vstack(Cn); cloth=np.concatenate(clo); F=np.vstack(Fn)
# orient bridging triangles consistently with the foot (fix winding by comparing to the ankle radial direction)
tri=V4[F]; nrm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); cen=tri.mean(1)
# compact: drop unreferenced vertices
usedv=np.zeros(len(V4),bool); usedv[F.ravel()]=True; remap=-np.ones(len(V4),int); remap[usedv]=np.arange(usedv.sum())
V4=V4[usedv]; C=C[usedv]; cloth=cloth[usedv]; F=remap[F]
print('feet swapped: verts',len(V4),'faces',len(F))
N=len(V4); A=adjacency(F,N); deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
wband=(1-smoothstep(np.abs(V4[:,1]-YC),0.012,0.04))*(np.abs(V4[:,0])>0.04)
V4=taubin(V4,wband,25)
# remove the inner layer of the old shorts hem (faces whose normal points back toward the leg axis)
_tri=V4[F]; _fn=np.cross(_tri[:,1]-_tri[:,0],_tri[:,2]-_tri[:,0]); _fn/=np.maximum(np.linalg.norm(_fn,axis=1,keepdims=True),1e-12); _c=_tri.mean(1)
_sg=np.sign(_c[:,0]); _rad=np.c_[_c[:,0]-_sg*0.125,np.zeros(len(F)),_c[:,2]+0.05]; _rad/=np.maximum(np.linalg.norm(_rad,axis=1,keepdims=True),1e-9)
_dead=(np.abs(_c[:,0])>0.02)&(_c[:,1]>0.5)&(_c[:,1]<0.8)&((_fn*_rad).sum(1)<-0.1)
print('inner hem faces removed',int(_dead.sum())); F=F[~_dead]
for _rep in range(2):
    nn=vnormals(V4,F); ns=A@nn; ns/=np.maximum(np.linalg.norm(ns,axis=1,keepdims=True),1e-9)
    bad=((nn*ns).sum(1)<0.55).astype(float)*(V4[:,1]<1.4)
    print('flipped-normal verts',int(bad.sum()))
    for _ in range(3): bad=np.clip(A@bad+bad,0,1)
    V4=taubin(V4,bad*0.9,40)
V=V4  # (mm / armness are recomputed after the torso refine below)
af=arm_fields(V4); darm=np.minimum.reduce([a[0] for a in af]+[a[2] for a in af]); armness=1-smoothstep(darm,0.075,0.115)
th=cKDTree(head.vertices); dh,_=th.query(V4); dfz,_=cKDTree(V4[dh<1e-5]).query(V4); mm=smoothstep(dfz,0.015,0.05)*(V4[:,1]<1.56)
# ---------- 6. refine the torso once so the muscle detail has resolution ----------
ext=np.c_[C,cloth]
cen=V4[F].mean(1)
sel=(cen[:,1]>0.93)&(cen[:,1]<1.47)&(np.abs(cen[:,0])<0.21)&(armness[F].mean(1)<0.6)
V5,F5,E5=refine(V4,F,ext,sel); print('torso refine: verts',len(V5),'faces',len(F5))
print('after 2nd refine: verts',len(V5),'faces',len(F5))
V,F=V5,F5; C=E5[:,:3]; cloth=np.clip(E5[:,3],0,1); N=len(V)
A=adjacency(F,N); deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
ptr=cKDTree(V[pinned_core[:0].sum()*0:][:0]) if False else None
# recompute fields on refined mesh
th=cKDTree(head.vertices); dh,_=th.query(V); dfz,_=cKDTree(V[dh<1e-5]).query(V)
mm=smoothstep(dfz,0.015,0.05)*(V[:,1]<1.56)
af=arm_fields(V); darm=np.minimum.reduce([a[0] for a in af]+[a[2] for a in af]); armness=1-smoothstep(darm,0.075,0.115)
# ---------- 7. muscle sculpt (displacement along vertex normals) ----------
wz=(1-armness)*mm*smoothstep(V[:,1],0.9,0.95)*(1-smoothstep(V[:,1],1.56,1.60))
V=taubin(V,np.clip(wz+0.5*spread(cloth.copy(),3)*mm,0,1),45)
n=vnormals(V,F); nz=n[:,2]; x=V[:,0]; yy=V[:,1]; ax=np.abs(x)
front=smoothstep(nz,0.15,0.65); back=smoothstep(-nz,0.15,0.65)
wt=(1-armness)*mm*smoothstep(yy,0.93,1.0)*(1-smoothstep(yy,1.42,1.5))
band=lambda y,a,b,e=0.012: smoothstep(y,a-e,a+e)*(1-smoothstep(y,b-e,b+e))
d=np.zeros(N)
# pectorals
for sg in (1,-1): d+=0.034*np.exp(-0.5*(((x-sg*0.088)/0.064)**2+((yy-1.32)/0.05)**2))*front
d+=-0.011*G(yy,1.258-0.012*np.minimum(ax/0.1,1.2),0.010)*smoothstep(ax,0.012,0.04)*(1-smoothstep(ax,0.115,0.15))*front
d+=-0.0045*G(x,0,0.007)*band(yy,1.255,1.40)*front
# abs
d+=-0.0050*G(x,0,0.0065)*band(yy,0.99,1.255)*front
for yk in (1.215,1.150,1.085): d+=-0.0045*G(yy,yk,0.0075)*(1-smoothstep(ax,0.055,0.085))*front
d+=0.0035*G(ax,0.038,0.03)*band(yy,1.05,1.24)*front
# iliac furrows (diagonal grooves toward the groin)
for sg in (1,-1):
    a_=np.array([sg*0.085,1.05]); b_=np.array([sg*0.045,0.965]); P=np.c_[x,yy]; ab=b_-a_; tt=np.clip(((P-a_)@ab)/(ab@ab),0,1); dist=np.linalg.norm(P-(a_+tt[:,None]*ab),axis=1)
    d+=-0.0055*np.exp(-0.5*(dist/0.009)**2)*front
# back: spine groove, scapulae, lats
d+=-0.0045*G(x,0,0.008)*band(yy,0.98,1.42)*back
for sg in (1,-1):
    d+=0.011*np.exp(-0.5*(((x-sg*0.095)/0.065)**2+((yy-1.33)/0.055)**2))*back
    d+=0.006*np.exp(-0.5*(((x-sg*0.15)/0.04)**2+((yy-1.2)/0.09)**2))*back
    d+=0.004*np.exp(-0.5*(((x-sg*0.07)/0.04)**2+((yy-1.05)/0.05)**2))*back   # erectors
# deltoid caps (outside arm tube handled by armness): add on the shoulder cap
for sg in (1,-1):
    dd_=np.linalg.norm(V-np.array([sg*0.255,1.40,-0.06]),axis=1); d+=0.007*np.exp(-0.5*(dd_/0.05)**2)*(1-smoothstep(yy,1.46,1.52))
dsp=d*np.maximum(wt,(armness>0.5)*mm*0.0+0)  # torso-only displacement
V6=V+(dsp)[:,None]*n
shade=np.clip(1+16*dsp,0.78,1.10)
# ---------- 8. skin colour field ----------
Tskin=np.array([100.,29.,16.])
sk=(cloth<0.5); Cs=np.where(sk[:,None],C,0.0); Wk=sk.astype(float)
# smooth the existing skin colours (they carry baked noise) and diffuse them into cloth areas
for _ in range(60):
    Cs=(A@Cs+Cs)/(deg[:,None]+1); Wk=(A@Wk+Wk)/(deg+1)
    Cs=np.where(sk[:,None],C*0.35+Cs*0.65*0+Cs*0.0+C*0.0+Cs*0.65,Cs) if False else Cs
local=Cs/np.maximum(Wk,1e-6)[:,None]
conf=np.clip(Wk*2.2,0,1)   # confidence of diffusion
fillc=conf[:,None]*local+(1-conf[:,None])*Tskin
# gently pull the diffused colour toward the target skin tone so torso/legs/arms read as one skin
fillc=0.85*fillc+0.15*Tskin
Cskin=np.where(cloth[:,None]>0.5,fillc,C)
Cskin=(1-np.clip(cloth*1.4,0,1))[:,None]*C+np.clip(cloth*1.4,0,1)[:,None]*fillc
tz=(1-armness)*smoothstep(yy,0.80,0.9)*(1-smoothstep(yy,1.52,1.58))*mm
rwt=np.clip(np.maximum(np.clip(cloth*1.4,0,1),tz),0,1)
Cskin=(1-rwt)[:,None]*C+rwt[:,None]*fillc
for _ in range(25): Cskin=(A@Cskin+Cskin)/(deg[:,None]+1)*np.where((mm>0.99)[:,None],1,0)+Cskin*np.where((mm>0.99)[:,None],0,1)
wn=(smoothstep(yy,1.36,1.52)*(1-armness)*mm)[:,None]*0.85
Cskin=(1-wn)*Cskin+wn*np.array([122.,36.,19.])
for _ in range(6): Cskin=(A@Cskin+Cskin)/(deg[:,None]+1)*np.where((mm>0.99)[:,None],1,0)+Cskin*np.where((mm>0.99)[:,None],0,1)
Cskin=Cskin*shade[:,None]
# ---------- 9. swimsuit: class field + local refinement for crisp colour edges ----------
SUIT=dict(brief=0,jammer=1)[VARIANT]
CONTOUR=float(os.environ.get('CONTOUR','0.024'))
# Olympic racing brief: low-rise waistband (WAIST), high-cut leg openings: slim side height (WAIST - yl at the hip) ~ SIDE_H
WAIST=float(os.environ.get('WAIST','0.958')); HEM_J=0.525
SIDE_H=float(os.environ.get('SIDE_H','0.060')); YL0=float(os.environ.get('YL0','0.795'))
# class colours: 1 royal/cobalt fabric, 2 slightly deeper waistband / slide sole, 3 white (slide trim only)
COBALT=np.array([float(x) for x in os.environ.get('COBALT','2,12,82').split(',')]); COBALT_D=COBALT*0.80; WHITE=np.array([196.,196.,198.])
def lin_leg(y,sg): return 0.115*sg+(0.135*sg-0.115*sg)*np.clip((0.88-y)/0.41,0,1)
def suit_class(P):
    x,y,z=P[:,0],P[:,1],P[:,2]; ax=np.abs(x); sg=np.where(x>=0,1,-1)
    if SUIT==0:
        # leg opening: crotch level YL0 at the midline rising to WAIST-SIDE_H at the hip (ax=0.20)
        k=(WAIST-SIDE_H-YL0)/(((0.20-0.02)/0.17)**1.25)
        yl=YL0+k*np.clip((ax-0.02)/0.17,0,1.3)**1.25
        # slightly lower cut at the back so the seat is covered
        yl=yl-0.010*(z<-0.04)
        inside=(y<WAIST)&(y>yl)&(ax<0.26)
        dl=y-yl
    else:
        inside=(y<WAIST)&(y>HEM_J)&(ax<0.26); dl=y-HEM_J
    cls=np.where(inside,1,0)
    # subtle deeper waistband (brief) / hem band
    if SUIT==0: cls=np.where(inside&(y>WAIST-0.012),2,cls)
    # swim slides: deep-blue sole with a white top edge, cobalt strap with white pin-stripes
    foot=(y<0.13)&(ax>0.05)&(ax<0.27)&(z>-0.17)
    sole=foot&(y<0.026); cls=np.where(sole,2,cls); cls=np.where(foot&(y>=0.026)&(y<0.031),3,cls)
    strap=foot&(y>=0.031)&(y<0.085)&(z>0.022)&(z<0.062); cls=np.where(strap,1,cls)
    cls=np.where(strap&((z<0.027)|(z>0.057)),3,cls)
    return cls
palette=np.array([[0,0,0],COBALT,COBALT_D,WHITE],float)
cls=suit_class(V)
for it in range(3):
    cen_cls=cls[F]; diff=(cen_cls.max(1)!=cen_cls.min(1))
    if diff.sum()==0: break
    ext=np.c_[Cskin,np.zeros(len(V)),shade]
    V,F,ex=refine(V6 if it==0 else V,F,ext if it==0 else np.c_[Cskin,np.zeros(len(V)),shade],diff)
    Cskin=ex[:,:3]; shade=ex[:,4]; V6=V
    cls=suit_class(V); print('suit refine',it,len(V),len(F),'boundary faces',diff.sum())
N=len(V)
V6=V
Cfinal=np.where(cls[:,None]>0,palette[cls]*np.clip(shade[:,None],0.9,1.05)**0,Cskin)
# slight thickness for the suit: smooth mask pushed outward along normals
A=adjacency(F,N); deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
sm=(cls>0).astype(float)
for _ in range(6): sm=(A@sm+sm)/(deg+1)
sm=np.clip((sm-0.15)/0.7,0,1)
n2=vnormals(V6,F)
V6=V6+(np.where(V6[:,1]<0.13,0.005,0.0025)*sm)[:,None]*n2
if SUIT==0:
    # modest natural front contour of a competition brief on an athlete: soft forward swell centred on the midline, fabric-covered, fades into the suit
    gx=np.exp(-(V6[:,0]/0.050)**2-((V6[:,1]-0.862)/0.055)**2)
    fw=smoothstep(n2[:,2],0.10,0.50)
    dvec=np.c_[0.4*n2[:,0],0.4*n2[:,1]-0.0,0.6*n2[:,2]+0.4]; dvec/=np.linalg.norm(dvec,axis=1,keepdims=True)
    V6=V6+(CONTOUR*gx*fw*sm)[:,None]*dvec
Cskin=Cfinal
out=trimesh.Trimesh(V6,F,vertex_colors=np.c_[np.clip(Cskin,0,255),np.full(N,255)].astype(np.uint8),process=False)
s.geometry['geometry_1']=out; s.export(OUT)
h0=trimesh.load(SRC).geometry['geometry_0'].bounds; h1=trimesh.load(OUT).geometry['geometry_0'].bounds
print('head bbox max diff mm',1000*np.abs(h0-h1).max()); print('body verts',N,'faces',len(F))
