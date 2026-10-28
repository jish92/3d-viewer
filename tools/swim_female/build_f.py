import sys,json
from common import *
from geom import *
import trimesh,os
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
# ---------------- suit fields (racing one-piece: scoop front, spaghetti straps, racerback, competition leg cut) ----------------
def interp(yy,pts): return np.interp(yy,[p[0] for p in pts],[p[1] for p in pts])
PAR=json.load(open('suit_params.json'))
NAVY=np.array(PAR['navy'],float)
# work on a once-subdivided copy of the body so clipped edges (scoop, armholes, leg lines) are smooth
Vq,Fq,(Vbq,)=subdivide(Vb,F,[Vb],PAR['sub']); Nq=len(Vq)
xq,yq,zq=Vq.T; axq=np.abs(xq); sgq=np.sign(xq); nrmq=vnormals(Vq,Fq)
zf=smoothstep(zq,-0.02,0.06)   # 1 front,0 back
# top edge: front = shallow scoop between the strap corners then a diagonal armhole line; back = racerback (armhole line rises toward the spine)
cx,cy=PAR['neck_corner_x'],PAR['neck_corner_y']
fr_pts=[(0.0,PAR['neck_center_y'])]+[tuple(p) for p in PAR['neck_mid']]+[tuple(p) for p in PAR['arm_front']]
top_f=np.interp(axq,[p[0] for p in fr_pts],[p[1] for p in fr_pts])
top_b=np.interp(axq,[p[0] for p in PAR['back_edge']],[p[1] for p in PAR['back_edge']])
top=zf*top_f+(1-zf)*top_b
f_top=(top-yq)*0.9
xe=interp(yq,[tuple(p) for p in PAR['armhole']])
f_arm=(xe-axq)*0.85
# leg openings
yc=PAR['leg_in']; yo=PAR['leg_out']; yfr=PAR['leg_front']
def legline(ax_,z_):
    # sporty beach-volleyball cut: very high leg lines (front diagonal to the iliac crest, back curving in to a narrow cheeky panel)
    Lf=np.interp(ax_,[p[0] for p in PAR['leg_front_curve']],[p[1] for p in PAR['leg_front_curve']])
    Lb=np.interp(ax_,[p[0] for p in PAR['leg_back_curve']],[p[1] for p in PAR['leg_back_curve']])
    zz=smoothstep(z_,-0.01,0.05)
    return zz*Lf+(1-zz)*Lb
Lv=legline(axq,zq)
f_leg=(yq-Lv)*0.95
fsuit=np.minimum.reduce([f_top,f_arm,f_leg])
# compression smoothing of suit surface
A=adjacency(Fq,Nq); deg=np.asarray(A.sum(1)).ravel()
Vs=Vq.copy()
for _ in range(PAR['smooth_it']):
    for lam in (0.5,-0.53): Vs=Vs+lam*((A@Vs)/deg[:,None]-Vs)
wsm=PAR['smooth_w']*(0.4+0.6*smoothstep(yq,1.15,1.25)*(1-smoothstep(yq,1.38,1.44))*smoothstep(zq,0.0,0.05))
Vsuit=Vq+(Vs-Vq)*wsm[:,None]
Vsuit=Vsuit+nrmq*PAR['thick']
Vc,Fc,(nc,Vbc),fc,used=clip_mesh(Vsuit,Fq,fsuit,[nrmq,Vq])
Csuit=np.tile(NAVY,(len(Vc),1))
# ---- spaghetti straps: ribbons following the body surface (front corner -> over the shoulder -> racerback junction)
from scipy.ndimage import gaussian_filter1d
kdq=cKDTree(Vq)
def surf(P,k=4):
    d,i=kdq.query(P,k=k); w=1/(d+1e-4); w/=w.sum(1,keepdims=True)
    n=(nrmq[i]*w[...,None]).sum(1); n/=np.linalg.norm(n,axis=1,keepdims=True)
    v=(Vq[i]*w[...,None]).sum(1)
    return P-((P-v)*n).sum(1,keepdims=True)*n, n
def pick(x0,y0,mode):
    m=(np.abs(xq-x0)<0.005)
    if mode=='front': m&=(np.abs(yq-y0)<0.005)&(zq>0.0)&(zq<0.25); return Vq[m][np.argmax(Vq[m][:,2])]
    if mode=='back':  m&=(np.abs(yq-y0)<0.005)&(zq<0.0)&(zq>-0.25); return Vq[m][np.argmin(Vq[m][:,2])]
    m&=(zq>-0.07)&(zq<0.05)&(yq<1.48)&(yq>1.3); return Vq[m][np.argmax(Vq[m][:,1])]
def catmull(P,n=140):
    P=np.vstack([2*P[0]-P[1],P,2*P[-1]-P[-2]]); out=[]
    for i in range(1,len(P)-2):
        for t in np.linspace(0,1,n//(len(P)-3),endpoint=False):
            t2,t3=t*t,t*t*t
            out.append(0.5*((2*P[i])+(-P[i-1]+P[i+1])*t+(2*P[i-1]-5*P[i]+4*P[i+1]-P[i+2])*t2+(-P[i-1]+3*P[i]-3*P[i+1]+P[i+2])*t3))
    out.append(P[-2]); return np.array(out)
def strap(sg):
    x1=PAR['strap_x'] if 'strap_x' in PAR else 0.09
    pts=[pick(x1,PAR['neck_corner_y']-0.028,'front'),pick(x1,PAR['neck_corner_y'],'front'),pick(x1,PAR['neck_corner_y']+0.03,'front'),
         pick(x1,0,'top'),pick(x1-0.012,1.41,'back'),pick(0.055,1.355,'back'),pick(0.026,1.318,'back'),pick(0.018,1.286,'back')]
    P=np.array(pts); P[:,0]=np.abs(P[:,0])
    C=catmull(P,160)
    for _ in range(2):
        C=np.c_[[gaussian_filter1d(C[:,k],2.0,mode='nearest') for k in range(3)]].T; C,_n=surf(C)
    C,n=surf(C)
    t=np.gradient(C,axis=0); t/=np.linalg.norm(t,axis=1,keepdims=True)
    bvec=np.cross(t,n); bvec/=np.linalg.norm(bvec,axis=1,keepdims=True)
    hw=PAR['strap_hw']; off=PAR['strap_off']
    rows=[]
    for s in (-1,0,1):
        Q,nq=surf(C+bvec*hw*s*1.0); rows.append(Q+nq*off)
    # taper last 1.5 cm into the panel (sink slightly) at both ends
    V=np.concatenate([np.stack(rows,1)]).reshape(-1,3)   # [sample,3] -> flat
    ns=len(C); idx=lambda i,j: i*3+j; F_=[]
    for i in range(ns-1):
        for j in range(2): F_+=[(idx(i,j),idx(i+1,j),idx(i+1,j+1)),(idx(i,j),idx(i+1,j+1),idx(i,j+1))]
    F_=np.array(F_); V0=V.copy()
    # outward winding
    fn=np.cross(V[F_[:,1]]-V[F_[:,0]],V[F_[:,2]]-V[F_[:,0]]); cn=V[F_].mean(1); _,nn=surf(cn)
    if (fn*nn).sum()<0: F_=F_[:,::-1]
    Vbs,_=surf(V); 
    if sg<0: V=V*np.array([-1,1,1]); Vbs=Vbs*np.array([-1,1,1]); F_=F_[:,::-1]
    return V,F_,Vbs
extraV=[];extraF=[];extraC=[];extraB=[]
off_=len(Vc)
for sg in (1,-1):
    V_,F_,B_=strap(sg); extraV.append(V_);extraF.append(F_+off_);extraC.append(np.tile(NAVY,(len(V_),1)));extraB.append(B_); off_+=len(V_)
# ---- small US flag patch (vertex colours; crisp: every stripe is its own strip of vertices)
RED_F=np.array(PAR.get('flag_red',[128,6,12]),float); WHITE_F=np.array(PAR.get('flag_white',[205,205,208]),float); BLUE_F=np.array(PAR.get('flag_blue',[6,12,70]),float)
kds=cKDTree(Vc)
nsu=vnormals(Vc,Fc)
def surf_s(P,k=6):
    d,i=kds.query(P,k=k); w=1/(d+1e-4); w/=w.sum(1,keepdims=True)
    n=(nsu[i]*w[...,None]).sum(1); n/=np.linalg.norm(n,axis=1,keepdims=True)
    v=(Vc[i]*w[...,None]).sum(1)
    return P-((P-v)*n).sum(1,keepdims=True)*n, n
FSIDE=float(os.environ.get('FLAG_SIDE','-1'))   # -1: image-left in the front view (as in the reference photo) = wearer's right chest
FW,FH=PAR['flag_w'],PAR['flag_h']
c0=np.array([FSIDE*abs(PAR['flag_cx']),PAR['flag_cy'],0.0])
# find the suit surface in front of the centre: highest-z suit vertex near (x,y)
m=(np.abs(Vc[:,0]-c0[0])<0.01)&(np.abs(Vc[:,1]-c0[1])<0.01)&(Vc[:,2]>0.05); c0=Vc[m][np.argmax(Vc[m][:,2])]
c0,nC=surf_s(c0[None]); c0=c0[0]; nC=nC[0]
uax=np.array([1.0,0,0]); uax-=nC*(uax@nC); uax/=np.linalg.norm(uax); vax=np.cross(nC,uax); vax*=np.sign(vax[1])
def FP(u,v): return c0+uax*(u-0.5)*FW+vax*(v-0.5)*FH      # u,v in [0,1], v up
fV=[];fC=[]
fF_list=[]
def rect(u0,u1,v0,v1,col,nu):
    base=sum(len(a) for a in fV); us=np.linspace(u0,u1,nu+1); P=np.array([FP(u,v) for v in (v0,v1) for u in us])
    fV.append(P); fC.append(np.tile(col,(len(P),1))); rl=nu+1
    for i in range(nu): fF_list.extend([(base+i,base+i+1,base+rl+i+1),(base+i,base+rl+i+1,base+rl+i)])
CANT_U=0.40; CANT_V0=1-7/13
for k in range(13):                       # k=0 top stripe (red)
    v1=1-k/13; v0=1-(k+1)/13; col=RED_F if k%2==0 else WHITE_F
    if k<7: rect(CANT_U,1.0,v0,v1,col,10)
    else: rect(0.0,1.0,v0,v1,col,16)
rect(0.0,CANT_U,CANT_V0,1.0,BLUE_F,8)
# stars: 50 -> staggered rows of 6 and 5 (9 rows), tiny diamonds on the canton
sw=0.00075
star_start=sum(len(a) for a in fV); nstars=0
for r_ in range(9):
    for c_ in range(6 if r_%2==0 else 5):
        uc=CANT_U*(0.0833+c_*0.1667+(0.0 if r_%2==0 else 0.0833)); vc=CANT_V0+(1-CANT_V0)*(0.07+0.86*r_/8.0)
        cpt=FP(uc,vc); du=uax*sw; dv=vax*sw
        base=sum(len(a) for a in fV); fV.append(np.array([cpt-dv,cpt+du,cpt+dv,cpt-du]))
        fC.append(np.tile(WHITE_F,(4,1))); fF_list.extend([(base,base+1,base+2),(base,base+2,base+3)]); nstars+=1
fVv=np.vstack(fV); fCv=np.vstack(fC); fFv=np.array(fF_list)
# conform every flag vertex to the suit surface (+offset); stars sit 0.3 mm higher to avoid z-fighting
Pf,nf=surf_s(fVv)
FV=Pf+nf*PAR['flag_off']
fn=np.cross(FV[fFv[:,1]]-FV[fFv[:,0]],FV[fFv[:,2]]-FV[fFv[:,0]]); nn=nf[fFv].mean(1)
flip=(fn*nn).sum(1)<0; fFv[flip]=fFv[flip][:,[0,2,1]]
# stars: the last vertices are the stars -> raise slightly
FV[star_start:]+=nf[star_start:]*0.0005
_,jj=kds.query(fVv); fB=Vbc[jj]
extraV.append(FV);extraF.append(fFv+off_);extraC.append(fCv);extraB.append(fB); off_+=len(FV)
Vc=np.vstack([Vc]+extraV); Fc=np.vstack([Fc]+extraF); Csuit=np.vstack([Csuit]+extraC); Vbc=np.vstack([Vbc]+extraB)
nc=np.vstack([nc]+[np.zeros((len(v),3)) for v in extraV]); fc=np.concatenate([fc]+[np.ones(len(v)) for v in extraV])
print('suit total verts',len(Vc),'tris',len(Fc),'flag tris',len(fFv),'flag centre',c0.round(3))
# skin body: drop faces whose verts are all well inside the suit (original body mesh)
xo,yo_,zo=Vb.T
# evaluate suit field on the original vertices to hide covered skin
zfo=smoothstep(zo,-0.02,0.06); axo=np.abs(xo); sgo=np.sign(xo)
topo=zfo*np.interp(axo,[p[0] for p in fr_pts],[p[1] for p in fr_pts])+(1-zfo)*np.interp(axo,[p[0] for p in PAR['back_edge']],[p[1] for p in PAR['back_edge']])
Lvo=legline(axo,zo)
fsuit_o=np.minimum.reduce([(topo-yo_)*0.9,(interp(yo_,[tuple(p) for p in PAR['armhole']])-axo)*0.85,(yo_-Lvo)*0.95])
deepin=fsuit_o>PAR['hide_depth']
keepF=~deepin[F].all(1)
Fskin=F[keepF]
Cskin=np.tile(SKIN,(N,1))
fsuit=fsuit_o
np.savez('stage_body.npz',Vb=Vb,F=F,Fskin=Fskin,Vc=Vc,Fc=Fc,Csuit=Csuit,Cskin=Cskin,fsuit=fsuit,fc=fc,Vbc=Vbc,nc=nc,y0=y0)
json.dump({k:v.tolist() for k,v in J.items()},open('joints_y.json','w'))
meshes=[]
m=trimesh.Trimesh(Vb,Fskin,process=False); m.visual.vertex_colors=np.c_[Cskin,np.full(N,255)].astype(np.uint8); meshes.append(m)
m=trimesh.Trimesh(Vc,Fc,process=False); m.visual.vertex_colors=np.c_[Csuit,np.full(len(Vc),255)].astype(np.uint8); meshes.append(m)
trimesh.Scene(meshes).export(OUT)
print('suit verts',len(Vc),'tris',len(Fc),'skin tris',len(Fskin))
