import sys,json
from common import *
from geom import *
import trimesh
from scipy.spatial import cKDTree
STAGE=sys.argv[1] if len(sys.argv)>1 else 'full'
OUT=sys.argv[2] if len(sys.argv)>2 else 'her_full.glb'
V0,polys,G,E=load()
Yall=yup(V0)
body=np.array(sorted(i for i,w in G['body'] if w>.5))
y0=Yall[body][:,1].min()
SH=np.array([0,y0,0])
Y=Yall-SH
J={}
for k,v in G.items():
    if k.startswith('joint-'): J[k[6:]]=Y[[i for i,w in v]].mean(0)
idx=-np.ones(len(V0),int); idx[body]=np.arange(len(body))
bp=[[idx[i] for i in p] for p in polys if all(idx[i]>=0 for i in p)]
Vb=Y[body].copy(); F=tri(bp); N=len(Vb)
NAVY=np.array([2,5,32.]);RED=np.array([128,4,10.]);WHITE=np.array([196,196,198.])
SKIN=np.array([190,112,78.])
x,y,z=Vb[:,0],Vb[:,1],Vb[:,2]; ax=np.abs(x); sg=np.sign(x)
nrm=vnormals(Vb,F)
# ---------------- suit fields ----------------
def interp(yy,pts): return np.interp(yy,[p[0] for p in pts],[p[1] for p in pts])
PAR=json.load(open('suit_params.json'))
# neckline (crew-high): front lower than back
zf=smoothstep(z,-0.02,0.06)   # 1 front,0 back
nl_front=PAR['neck_front']+PAR['neck_w']*(ax/0.07)**2
nl_back=PAR['neck_back']+PAR['neck_w']*(ax/0.07)**2
nl=zf*nl_front+(1-zf)*nl_back
f_neck=(nl-y)*0.9
xe=interp(y,[tuple(p) for p in PAR['armhole']])
f_arm=(xe-ax)*0.85
# leg openings
yc=PAR['leg_in']; yo=PAR['leg_out']; yfr=PAR['leg_front']
lx=0.115*sg; dx=(x-lx)*sg; dz=z-0.02; r=np.maximum(np.hypot(dx,dz),1e-6)
uo=dx/r; uf=dz/r
Lv=yc+(yo-yc)*(0.5+0.5*uo)+yfr*np.maximum(uf,0)-PAR['leg_back']*np.maximum(-uf,0)
# near the centre line the 'outward' direction vanishes -> inner level
cen=smoothstep(ax,0.0,0.05); Lv=yc+(Lv-yc)*cen
f_leg=(y-Lv)*0.95
fsuit=np.minimum.reduce([f_neck,f_arm,f_leg])
fsuit=np.where(y<1.0,np.minimum(f_leg,f_arm),fsuit) if False else fsuit
# compression smoothing of suit surface
A=adjacency(F,N); deg=np.asarray(A.sum(1)).ravel()
Vs=Vb.copy()
for _ in range(PAR['smooth_it']):
    for lam in (0.5,-0.53): Vs=Vs+lam*((A@Vs)/deg[:,None]-Vs)
wsm=PAR['smooth_w']*(0.4+0.6*smoothstep(y,1.15,1.25)*(1-smoothstep(y,1.38,1.44))*smoothstep(z,0.0,0.05))
Vsuit=Vb+(Vs-Vb)*wsm[:,None]
Vsuit=Vsuit+nrm*PAR['thick']
# clip suit
attrs=[nrm,Vb]
Vc,Fc,(nc,Vbc),fc,used=clip_mesh(Vsuit,F,fsuit,[nrm,Vb])
# side line z(y) from lateral-facing vertices
lat=(np.abs(nrm[:,0])>0.92)&(y>0.7)&(y<1.4)&(ax<0.3)
bins=np.arange(0.7,1.41,0.05); zl=[]
for a,b in zip(bins[:-1],bins[1:]):
    m=lat&(y>=a)&(y<b); zl.append(z[m].mean() if m.sum() else np.nan)
zl=np.array(zl); ok=~np.isnan(zl); ZS=((bins[:-1]+0.025)[ok],zl[ok]); print('side line z',zl.round(3))
# colours: classes by distance to boundary
def classify(P,fv,nv):
    cls=np.zeros(len(P),int)+1  # navy
    # edge trims: white outer stripe then red band (per boundary type approximated by fv)
    tw,tr=PAR['trim_w'],PAR['trim_r']
    cls=np.where(fv<tr,2,cls); cls=np.where(fv<tw,3,cls)
    # side panel
    inside=(fv>tr+0.004)
    sidep=(np.abs(nv[:,0])>0.5)
    ang=np.abs(P[:,2]-np.interp(P[:,1],ZS[0],ZS[1]))
    sc=inside&sidep&(P[:,1]<PAR['stripe_top'])
    cls=np.where(sc&(ang<PAR['stripe_w'][1]),3,cls); cls=np.where(sc&(ang<PAR['stripe_w'][0]),2,cls)
    return cls
cls=classify(Vc,fc,nc)
# refine boundaries of classes by midpoint subdivision where classes differ (for crisp lines)
def refine_cls(V_,F_,fv,nv,Vb_,iters=2):
    for it in range(iters):
        c=classify(V_,fv,nv); d=(c[F_].max(1)!=c[F_].min(1))
        if d.sum()==0: break
        # split longest edge of each differing triangle's all edges (1->4 split on flagged tris, with neighbours handled by conformity pass skipped: use full midpoint on all tri edges of flagged tris)
        E=np.concatenate([F_[d][:,[0,1]],F_[d][:,[1,2]],F_[d][:,[2,0]]]); E.sort(1); E=np.unique(E,axis=0)
        mp=(V_[E[:,0]]+V_[E[:,1]])/2; mf=(fv[E[:,0]]+fv[E[:,1]])/2; mn=(nv[E[:,0]]+nv[E[:,1]])/2; mb=(Vb_[E[:,0]]+Vb_[E[:,1]])/2
        key={tuple(e):len(V_)+i for i,e in enumerate(E)}
        V_=np.vstack([V_,mp]);fv=np.concatenate([fv,mf]);nv=np.vstack([nv,mn]);Vb_=np.vstack([Vb_,mb])
        out=[]
        for t in F_:
            e01=key.get(tuple(sorted((t[0],t[1]))));e12=key.get(tuple(sorted((t[1],t[2]))));e20=key.get(tuple(sorted((t[2],t[0]))))
            cnt=(e01 is not None)+(e12 is not None)+(e20 is not None)
            a,b,c_=t
            if cnt==0: out.append((a,b,c_))
            elif cnt==3: out+= [(a,e01,e20),(b,e12,e01),(c_,e20,e12),(e01,e12,e20)]
            elif cnt==1:
                if e01 is not None: out+=[(a,e01,c_),(e01,b,c_)]
                elif e12 is not None: out+=[(b,e12,a),(e12,c_,a)]
                else: out+=[(c_,e20,b),(e20,a,b)]
            else:
                if e01 is None: out+=[(b,e12,a),(e12,e20,a),(e12,c_,e20)]
                elif e12 is None: out+=[(c_,e20,b),(e20,e01,b),(e20,a,e01)]
                else: out+=[(a,e01,c_),(e01,e12,c_),(e01,b,e12)]
        F_=np.array(out)
    return V_,F_,fv,nv,Vb_
Vc,Fc,fc,nc,Vbc=refine_cls(Vc,Fc,fc,nc,Vbc,PAR['refine'])
cls=classify(Vc,fc,nc)
pal=np.array([[0,0,0],NAVY,RED,WHITE])
Csuit=pal[cls]
# skin body: drop faces whose verts are all well inside the suit
deepin=fsuit>PAR['hide_depth']
keepF=~deepin[F].all(1)
Fskin=F[keepF]
Cskin=np.tile(SKIN,(N,1))
np.savez('stage_body.npz',Vb=Vb,F=F,Fskin=Fskin,Vc=Vc,Fc=Fc,Csuit=Csuit,Cskin=Cskin,fsuit=fsuit,fc=fc,Vbc=Vbc,nc=nc,y0=y0)
json.dump({k:v.tolist() for k,v in J.items()},open('joints_y.json','w'))
meshes=[]
m=trimesh.Trimesh(Vb,Fskin,process=False); m.visual.vertex_colors=np.c_[Cskin,np.full(N,255)].astype(np.uint8); meshes.append(m)
m=trimesh.Trimesh(Vc,Fc,process=False); m.visual.vertex_colors=np.c_[Csuit,np.full(len(Vc),255)].astype(np.uint8); meshes.append(m)
trimesh.Scene(meshes).export(OUT)
print('suit verts',len(Vc),'tris',len(Fc),'skin tris',len(Fskin))
