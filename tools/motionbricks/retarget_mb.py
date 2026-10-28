"""Retarget MotionBricks SOMA output (Z-up world, 77 joints) onto the 17-bone swim rig. numpy only -> mb_clip.npz"""
import sys,pickle,numpy as np; sys.path.insert(0,'/tmp')
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import uniform_filter1d
exec(open('/tmp/fk.py').read())
src,out=sys.argv[1],sys.argv[2]
SIDE_LOOP=True
sk=np.load('/tmp/soma_skel.npz'); names=list(sk['names']); par=sk['par']; ref=sk['ref']; J=names.index
rest=pickle.load(open('/tmp/our_rest.pkl','rb'))
BONES=['Hips','Spine','Chest','Neck','Head','UpperArm.L','LowerArm.L','Hand.L','UpperArm.R','LowerArm.R','Hand.R','UpperLeg.L','LowerLeg.L','Foot.L','UpperLeg.R','LowerLeg.R','Foot.R']
anim=np.load(src)[::2]            # 60 Hz updates -> 30 Hz
N=len(anim); print('frames',N)
Pa,Qa=fk(anim,par); Pr,Qr=fk(ref,par)
C2=np.array([[0,1,0],[-1,0,0],[0,0,1.]])        # SOMA anim (Z up, +X fwd, +Y left) -> Blender rig frame (x left, -y fwd, z up)
C1=np.array([[1,0,0],[0,0,-1],[0,1,0.]])        # SOMA ref (Y up, +Z fwd, +X left)   -> rig frame
Pa=Pa@C2.T
Ra=Rot.from_quat(Qa.reshape(-1,4)).as_matrix().reshape(N,77,3,3)
Rr=Rot.from_quat(Qr).as_matrix()
def delta(j): return C2@np.einsum('fab,bc->fac',Ra[:,J(j)],Rr[J(j)].T)@C1.T   # world delta of SOMA joint vs its rest
RB={b:rest[b]['M'][:3,:3] for b in BONES}; HB={b:rest[b]['head'] for b in BONES}; PAR={b:rest[b]['parent'] for b in BONES}
dsrc={'Hips':'Hips','Spine':'Spine2','Chest':'Chest','Neck':'Neck1','Head':'Head'}
Rw={b:np.einsum('fab,bc->fac',delta(s),RB[b]) for b,s in dsrc.items()}
def unit(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
def swing(d0,d1):
    ax=np.cross(d0,d1); s=np.linalg.norm(ax,axis=-1,keepdims=True); c=(d0*d1).sum(-1,keepdims=True)
    ang=np.arctan2(s,c); ax=ax/np.maximum(s,1e-9); return Rot.from_rotvec(ax*ang).as_matrix()
# hips translation: pelvis centre from the two hip joints, vertical scaled by leg-length ratio, horizontal path removed
k=(0.88-0.10)/(0.929-0.072)
hip=(Pa[:,J('LeftLeg')]+Pa[:,J('RightLeg')])/2
hxy=hip[:,:2]; win=21
sway=hxy-uniform_filter1d(hxy,win,axis=0,mode='nearest')
tgt=np.zeros((N,3)); tgt[:,:2]=HB['Hips'][:2]+sway*k
tgt[:,2]=0.88+0.02+(hip[:,2]-Pr[J('LeftLeg'),1])*k

# leg IK: ankle placed at hip + k*(soma ankle - soma hip) so planted feet stay on one height, knee pole from the SOMA knee
def leg_ik(side,sd):
    hb,kb,ab=f'UpperLeg.{sd}',f'LowerLeg.{sd}',f'Foot.{sd}'
    Hj=tgt+np.einsum('fab,b->fa',Rw['Hips'],RB['Hips'].T@(HB[hb]-HB['Hips']))
    L1=np.linalg.norm(HB[kb]-HB[hb]); L2=np.linalg.norm(HB[ab]-HB[kb])
    hs,ks,as_=Pa[:,J(side+'Leg')],Pa[:,J(side+'Shin')],Pa[:,J(side+'Foot')]
    T=Hj+k*(as_-hs)
    v=T-Hj; dd=np.linalg.norm(v,axis=1); u=v/dd[:,None]; dd=np.minimum(dd,(L1+L2)*0.9995); 
    pole=(ks-hs)-((ks-hs)*u).sum(1,keepdims=True)*u; pole=unit(pole)
    a=(L1**2-L2**2+dd**2)/(2*dd); h=np.sqrt(np.maximum(L1**2-a**2,1e-8))
    knee=Hj+a[:,None]*u+h[:,None]*pole
    return Hj,knee,Hj+dd[:,None]*u,T
LEGD={}
for side,sd in (('Left','L'),('Right','R')):
    Hj,knee,ank,T=leg_ik(side,sd); LEGD[sd]=(Hj,knee,ank)
aim={'UpperArm.L':('LeftArm','LeftForeArm'),'LowerArm.L':('LeftForeArm','LeftHand'),'Hand.L':('LeftHand','LeftHandMiddleEnd'),
     'UpperArm.R':('RightArm','RightForeArm'),'LowerArm.R':('RightForeArm','RightHand'),'Hand.R':('RightHand','RightHandMiddleEnd'),
     'UpperLeg.L':('LeftLeg','LeftShin'),'LowerLeg.L':('LeftShin','LeftFoot'),'Foot.L':('LeftFoot','LeftToeBase'),
     'UpperLeg.R':('RightLeg','RightShin'),'LowerLeg.R':('RightShin','RightFoot'),'Foot.R':('RightFoot','RightToeBase')}
for b in BONES:
    if b in aim:
        p=PAR[b]; R0=np.einsum('fab,bc,cd->fad',Rw[p],RB[p].T,RB[b])
        d0=R0[:,:,1]; d1=unit(Pa[:,J(aim[b][1])]-Pa[:,J(aim[b][0])])
        if b.startswith('UpperLeg'): d1=unit(LEGD[b[-1]][1]-LEGD[b[-1]][0])
        if b.startswith('LowerLeg'): d1=unit(LEGD[b[-1]][2]-LEGD[b[-1]][1])
        Rw[b]=np.einsum('fab,fbc->fac',swing(d0,d1),R0)
# local pose rotations
Q={}
for b in BONES:
    p=PAR[b]
    if p is None: M=np.einsum('ab,fbc->fac',RB[b].T,Rw[b])
    else: M=np.einsum('ab,bc,fcd,fde->fae',RB[b].T,RB[p],np.linalg.inv(Rw[p]),Rw[b])
    Q[b]=M
Qm=np.stack([Q[b] for b in BONES],1)   # [N,17,3,3]
H=tgt.copy()
LOOP=len(sys.argv)>3 and sys.argv[3]=='loop'
if LOOP:
    # pick start/end frames in the straight sections (heading identical) with the closest pose + velocity, then crossfade the seam
    K=8
    feat=np.concatenate([Qm.reshape(N,-1),(H[:,2:3]-0.87)*3],1)
    vel=np.gradient(feat,axis=0)*6
    F=np.concatenate([feat,vel],1)
    best=(1e9,0,0)
    for s_ in range(int(0.9*30),int(1.6*30)):
        for e_ in range(int(7.8*30),N-1):
            d=np.linalg.norm(F[s_]-F[e_])
            if d<best[0]: best=(d,s_,e_)
    d,s_,e_=best; print('loop',s_,e_,'dist',round(d,3),'len s',(e_-s_)/30)
    Ql=Rot.from_matrix(Qm.reshape(-1,3,3)).as_quat().reshape(N,17,4)
    seg=np.arange(s_,e_)            # loop frames s..e-1 ; wrap goes e-1 -> s
    Qs=Ql[seg].copy(); Hs=H[seg].copy()
    for i in range(K):
        w=(i+1)/K; w=w*w*(3-2*w)
        fa=e_-K+i; fb=s_-K+i
        qa=Ql[fa]; qb=Ql[fb]; qb=qb*np.sign((qa*qb).sum(-1,keepdims=True))
        q=qa*(1-w)+qb*w; q/=np.linalg.norm(q,axis=-1,keepdims=True)
        Qs[len(seg)-K+i]=q; Hs[len(seg)-K+i]=H[fa]*(1-w)+H[fb]*w
    Qm=Rot.from_quat(Qs.reshape(-1,4)).as_matrix().reshape(len(seg),17,3,3); H=Hs
np.savez(out,Qm=Qm,hips=H,bones=np.array(BONES),Pa=Pa,hip=hip)
print('hips z range',tgt[:,2].min(),tgt[:,2].max())
