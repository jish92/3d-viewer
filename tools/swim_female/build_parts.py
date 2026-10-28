import sys,json
from common import *
from geom import *
import trimesh
from scipy.spatial import ConvexHull
D=np.load('stage_body.npz'); Vb=D['Vb'];F=D['F'];Fskin=D['Fskin'];Vc=D['Vc'];Fc=D['Fc'];Csuit=D['Csuit'];Cskin=D['Cskin'];y0=float(D['y0'])
V0,polys,G,E=load(); SHV=np.array([0,y0,0])
N=len(Vb); x,y,z=Vb.T
nrm=vnormals(Vb,F)
parts=[]   # (name,V,F,C)
HAIR=np.array([30,12,5.]); BROW=np.array([22,9,4.])
# ---------------- skin tweaks: lips, cheeks ----------------
C=Cskin.copy()
mouth=np.array(json.load(open('joints_y.json'))['mouth'])
lip=np.exp(-(((x)/0.024)**2+((y-(mouth[1]-0.002))/0.0075)**2+((z-0.105)/0.05)**2))
lipc=np.array([165,60,50.])
C=C*(1-lip[:,None]*0.85)+lipc*lip[:,None]*0.85
# ---------------- eyes ----------------
EV=yup(E['eyes_V'])-SHV
eyes=[]
for s in (1,-1):
    m=EV[:,0]*s>0; c=EV[m].mean(0); r=(EV[m].max(0)-EV[m].min(0)).max()/2
    iv,ifa=icosphere(4)
    P=c+iv*r*0.97
    ang=np.arccos(np.clip(iv[:,2],-1,1))   # from +z
    col=np.tile([212,201,191.],(len(iv),1))
    iris=np.array([46,19,7.]); pup=np.array([4,4,4.])
    limb=smoothstep(ang,0.40,0.50)    # 0 inside iris
    col=np.where((ang<0.48)[:,None],iris*(0.75+0.5*smoothstep(ang,0.20,0.40))[:,None],col)
    col=np.where((ang<0.46)[:,None]&(ang>0.40)[:,None],iris*0.5,col)
    col=np.where((ang<0.19)[:,None],pup,col)
    # shade sclera slightly toward edges
    parts.append(('eye',P,ifa,col)); print('eye',c.round(3),r.round(4))
# ---------------- brows ----------------
BV=yup(E['brow_V'])-SHV; BF=tri(list(E['brow_P']))
parts.append(('brow',BV+np.array([0,0,0.0005]),BF,np.tile(BROW,(len(BV),1))))
# ---------------- hair: scalp shell ----------------
HL=[(0.20,1.657),(0.11,1.657),(0.075,1.650),(0.0,1.648),(-0.012,1.62),(-0.024,1.55),(-0.034,1.50),(-0.05,1.45),(-0.3,1.46)]
zz=np.array([p[0] for p in HL])[::-1]; yy=np.array([p[1] for p in HL])[::-1]
hl=np.interp(z,zz,yy)+0.008-0.026*smoothstep(np.abs(x),0.012,0.07)*smoothstep(z,0.03,0.10)-0.03*smoothstep(np.abs(x),0.06,0.085)*smoothstep(z,0.06,0.12)-0.052*smoothstep(np.abs(x),0.050,0.078)*smoothstep(z,-0.06,-0.02)*(1-smoothstep(z,0.08,0.12))
fh=(y-hl)*0.9
fh=np.where(y<1.45,-1,fh)
fh=np.minimum(fh,(0.10-np.abs(x))*0.8+0.0) if False else fh
# bun placement
BUN=np.array([0,1.718,0.002]); BR=0.050
d_b=np.linalg.norm(Vb-BUN,axis=1)
off=0.0075+0.010*np.exp(-(d_b/0.09)**2)
Vh=Vb+nrm*off[:,None]
Vch,Fch,(nh,Vbh),fhc,_=clip_mesh(Vh,F,fh,[nrm,Vb])
# taper thickness to 0.0015 at the hairline
tap=smoothstep(fhc,0.0,0.012)
Vch=Vbh+nh*(0.0015+ (np.linalg.norm(Vbh-BUN,axis=1)*0)+ (0.0075-0.0015+0.010*np.exp(-(np.linalg.norm(Vbh-BUN,axis=1)/0.09)**2))*tap)[:,None]
# strand shading: streaks running back over the scalp
ang=np.arctan2(Vbh[:,0],-(Vbh[:,2]-BUN[2]))
sh=0.80+0.35*(0.5+0.5*np.sin(ang*38))*smoothstep(fhc,0.0,0.02)+0.15*smoothstep(fhc,0,0.05)*np.sin(ang*17+3)
Ch=HAIR*sh[:,None]*(1.0+0.35*smoothstep(Vbh[:,1],1.66,1.72)[:,None])
parts.append(('hair',Vch,Fch,Ch))
# ---------------- bun ----------------
bv,bf=icosphere(4)
th=np.arccos(np.clip(bv[:,1],-1,1)); ph=np.arctan2(bv[:,2],bv[:,0])
rad=BR*(1+0.05*np.sin(5*ph+4*th))*(1.0-0.12*smoothstep(-bv[:,1],0.5,1.0))   # slightly flattened toward the base
Pb=BUN+bv*rad[:,None]*np.array([1.0,0.88,1.0])[None]
shb=0.75+0.45*(0.5+0.5*np.sin(5*ph+4*th+1.0))
parts.append(('bun',Pb,bf,HAIR*shb[:,None]*1.15))
# ---------------- slides ----------------
RED=np.array([128,4,10.]);WHITE=np.array([196,196,198.]);NAVY=np.array([2,5,32.])
def hull_poly(P2,n=72):
    h=ConvexHull(P2); poly=P2[h.vertices]; c=poly.mean(0)
    # resample polygon by angle
    a=np.arctan2(poly[:,1]-c[1],poly[:,0]-c[0]); o=np.argsort(a); poly=poly[o]; a=a[o]
    out=[]
    tt=np.linspace(-np.pi,np.pi,n,endpoint=False)
    for t in tt:
        # ray from centre; intersect with polygon edges
        d=np.array([np.cos(t),np.sin(t)]); best=0
        for i in range(len(poly)):
            p1=poly[i]-c; p2=poly[(i+1)%len(poly)]-c
            M=np.array([d,-(p2-p1)]).T
            if abs(np.linalg.det(M))<1e-12: continue
            s,u=np.linalg.solve(M,p1)
            if 0<=u<=1 and s>0: best=max(best,s)
        out.append(c+d*best)
    return np.array(out),c
for s in (1,-1):
    fm=(y<0.06)&(x*s>0.05)
    pts=np.c_[x[fm],z[fm]]
    poly,c=hull_poly(pts)
    rings=[(0.004,0.0035,NAVY),(0.0105,0.0005,WHITE),(0.0125,-0.0045,WHITE),(0.0125,-0.0065,NAVY),(0.0120,-0.015,NAVY),(0.0105,-0.0215,NAVY),(0.0065,-0.0245,NAVY)]
    Vs=[];Fs=[];Cs=[]
    n=len(poly)
    # outward normals in plane (from centre)
    dirs=poly-c; dl=np.linalg.norm(dirs,axis=1,keepdims=True); dirs=dirs/dl
    # toe gets a little extra length
    for off,yy,col in rings:
        ex=off+0.004*np.maximum(dirs[:,1],0)
        P=np.c_[(poly+dirs*ex[:,None])[:,0],np.full(n,yy),(poly+dirs*ex[:,None])[:,1]]
        Vs.append(P); Cs.append(np.tile(col,(n,1)))
    k=len(rings)
    for r in range(k-1):
        for i in range(n):
            a=r*n+i;b=r*n+(i+1)%n;c2=(r+1)*n+i;d=(r+1)*n+(i+1)%n
            Fs+= [(a,c2,b),(b,c2,d)]
    Vs=np.vstack(Vs); base=len(Vs)
    # top cap & bottom cap fans
    Vs=np.vstack([Vs,[c[0],0.0035,c[1]],[c[0],-0.0245,c[1]]]); WHITE_=WHITE
    for i in range(n):
        Fs.append((base,i,(i+1)%n)); Fs.append((base+1,(k-1)*n+(i+1)%n,(k-1)*n+i))
    Cs=np.vstack(Cs+[NAVY[None],NAVY[None]])
    Fs=np.array(Fs)
    # make winding outward: check mean normal vs outward from centre
    pm=trimesh.Trimesh(Vs,Fs,process=False)
    if pm.volume<0: Fs=Fs[:,::-1]
    parts.append(('sole',Vs,Fs,Cs))
    # strap
    zc=0.115; hw=0.020
    fv=np.minimum(hw-np.abs(z-zc),y-0.006)
    sel=(y<0.14)&(x*s>0.05); fsel=sel[F].all(1)
    Fsub=F[fsel]; used=np.unique(Fsub); mp=-np.ones(N,int); mp[used]=np.arange(len(used))
    Vp=Vb[used]+nrm[used]*0.0035; Fp=mp[Fsub]
    Vp,Fp,(np_,)=subdivide(Vp,Fp,[nrm[used]],3)
    Vp,Fp,(np_,)=subdivide(Vp,Fp,[np_],0)
    fv=np.minimum(hw-np.abs(Vp[:,2]-zc),Vp[:,1]-0.006)
    Vst,Fst,(nst,),fst,_=clip_mesh(Vp,Fp,fv,[np_])
    zst=Vst[:,2]
    cst=np.where((np.abs(zst-zc)>hw-0.0055)[:,None],WHITE,RED)
    parts.append(('strap',Vst,Fst,cst))
# ---------------- assemble ----------------
LIFT=0.0245
allV=[];allF=[];allC=[];name=[];off=0
def add(nm,Vp,Fp,Cp):
    global off
    allV.append(Vp);allF.append(Fp+off);allC.append(Cp);name.extend([nm]*len(Vp));off+=len(Vp)
add('skin',Vb,Fskin,C)
add('suit',Vc,Fc,Csuit)
for nm,Vp,Fp,Cp in parts: add(nm,Vp,Fp,Cp)
V=np.vstack(allV);F2=np.vstack(allF);Cc=np.vstack(allC);name=np.array(name)
V=V+np.array([0,LIFT,0])
# remove unused verts
np.savez('her_parts.npz',V=V,F=F2,C=Cc,part=name)
m=trimesh.Trimesh(V,F2,process=False); m.visual.vertex_colors=np.c_[np.clip(Cc,0,255),np.full(len(V),255)].astype(np.uint8)
m.export('her_full.glb'); print('verts',len(V),'tris',len(F2),'height',V[:,1].max())
