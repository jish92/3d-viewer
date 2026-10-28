"""[female variant: documented female-gait adjustments, see gait_params.json "mb"; her geometry via /tmp/our_rest.pkl] Retarget MotionBricks SOMA output (Z-up world, 77 joints) onto the 17-bone swim rig. numpy only -> mb_clip.npz"""
import sys,pickle,json,os,numpy as np; sys.path.insert(0,'/tmp')
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

FEM=json.load(open('/workspace/me3d/swim_female/gait_params.json'))['mb']
if os.environ.get('FEM_OFF'): FEM={k_:(0 if k_ not in ('stride','pelvis_yaw_gain') else 1) for k_ in FEM}
HBL=rest['UpperLeg.L']['head'][2]; AYL=rest['Foot.L']['head'][2]
k=(HBL-AYL)/(0.929-0.072); print('k',k)
def rz(a): 
    c,s_=np.cos(a),np.sin(a); R=np.zeros(a.shape+(3,3)); R[...,0,0]=c;R[...,0,1]=-s_;R[...,1,0]=s_;R[...,1,1]=c;R[...,2,2]=1; return R
def axang(ax,a): return Rot.from_rotvec(ax*a[:,None]).as_matrix()
zup=np.array([0,0,1.])
# ---- straight-ahead lock: remove the slowly varying heading (low-pass of the pelvis yaw, window = one stride) from every world rotation and from the world positions
_lv=Pa[:,J('LeftLeg')]-Pa[:,J('RightLeg')]; _lv[:,2]=0
_yw=np.unwrap(np.arctan2(_lv[:,1],_lv[:,0])); HEAD=uniform_filter1d(_yw,21,mode='nearest'); HEAD=HEAD-np.median(HEAD)
print('heading removed: range deg %.2f..%.2f'%(np.degrees(HEAD.min()),np.degrees(HEAD.max())))
_hp=(Pa[:,J('LeftLeg')]+Pa[:,J('RightLeg')])/2; _hs=uniform_filter1d(_hp[:,:2],21,axis=0,mode='nearest')
def _rzp(a): 
    c,s_=np.cos(a),np.sin(a); R=np.zeros(a.shape+(3,3)); R[...,0,0]=c;R[...,0,1]=-s_;R[...,1,0]=s_;R[...,1,1]=c;R[...,2,2]=1; return R
_RH=_rzp(-HEAD)
Pa=Pa.copy(); _rel=Pa[:,:,:2]-_hs[:,None,:]
Pa[:,:,:2]=np.einsum('fab,fjb->fja',_RH[:,:2,:2],_rel)+_hs[:,None,:]
for _b in list(Rw): Rw[_b]=np.einsum('fab,fbc->fac',_RH,Rw[_b])
latv=Pa[:,J('LeftLeg')]-Pa[:,J('RightLeg')]; latv[:,2]=0; lat=unit(latv); fwd=np.cross(lat,zup)
yaw=np.unwrap(np.arctan2(lat[:,1],lat[:,0])); osc=yaw-uniform_filter1d(yaw,21,mode='nearest')
shv=Pa[:,J('LeftArm')]-Pa[:,J('RightArm')]; shv[:,2]=0; syaw=np.unwrap(np.arctan2(shv[:,1],shv[:,0])); sosc=syaw-uniform_filter1d(syaw,21,mode='nearest')
zl=Pa[:,J('LeftFoot'),2]; zr=Pa[:,J('RightFoot'),2]
swd=uniform_filter1d(np.clip((zl-zr)/0.12,-1,1),3,mode='nearest')
eps=(FEM['pelvis_yaw_gain']-1)*osc
th=np.radians(FEM['pelvis_tilt_extra_deg'])*swd           # + = left hip down
def stats(tag,Rh,Rsp,Rch,anx,anz=None):
    pass
Rh0=Rw['Hips'].copy(); Rc0=Rw['Chest'].copy()
def yaw_of(R): return np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]))
def roll_amp(R): # tilt of the pelvis lateral axis (bone x) out of the horizontal plane
    return np.degrees(np.arcsin(np.clip(R[:,2,0],-1,1)))
Rw['Hips']=np.einsum('fab,fbc->fac',rz(eps),Rw['Hips'])
Rw['Hips']=np.einsum('fab,fbc->fac',axang(fwd,-th),Rw['Hips'])
for b,cy,cr in (('Spine',FEM['spine_counter_yaw'],FEM['spine_counter_roll']),('Chest',FEM['chest_counter_yaw'],FEM['chest_counter_roll'])):
    Rw[b]=np.einsum('fab,fbc->fac',rz(-cy*eps),Rw[b]); Rw[b]=np.einsum('fab,fbc->fac',axang(fwd,cr*th),Rw[b])
print('pelvis yaw osc amplitude deg: before %.2f after %.2f; shoulder-line osc %.2f; pelvic tilt (rolled lateral axis) amp deg: before %.2f after %.2f'%(
   np.degrees(np.percentile(np.abs(osc),95)),np.degrees(np.percentile(np.abs(yaw_of(Rw['Hips'])-uniform_filter1d(np.unwrap(yaw_of(Rw['Hips'])),21,mode='nearest')),95)),np.degrees(np.percentile(np.abs(sosc),95)),
   np.percentile(np.abs(roll_amp(Rh0)),95),np.percentile(np.abs(roll_amp(Rw['Hips'])),95)))
# hips translation: pelvis centre from the two hip joints, vertical scaled by leg-length ratio, horizontal path removed
hip=(Pa[:,J('LeftLeg')]+Pa[:,J('RightLeg')])/2
hxy=hip[:,:2]; win=21
sway=hxy-uniform_filter1d(hxy,win,axis=0,mode='nearest')
tgt=np.zeros((N,3)); tgt[:,:2]=HB['Hips'][:2]+sway*k
tgt[:,2]=HBL+0.02+(hip[:,2]-Pr[J('LeftLeg'),1])*k

# leg IK: ankle placed at hip + k*(soma ankle - soma hip) so planted feet stay on one height, knee pole from the SOMA knee
def leg_ik(side,sd):
    hb,kb,ab=f'UpperLeg.{sd}',f'LowerLeg.{sd}',f'Foot.{sd}'
    Hj=tgt+np.einsum('fab,b->fa',Rw['Hips'],RB['Hips'].T@(HB[hb]-HB['Hips']))
    L1=np.linalg.norm(HB[kb]-HB[hb]); L2=np.linalg.norm(HB[ab]-HB[kb])
    hs,ks,as_=Pa[:,J(side+'Leg')],Pa[:,J(side+'Shin')],Pa[:,J(side+'Foot')]
    Hj0=tgt+np.einsum('fab,b->fa',Rh0,RB['Hips'].T@(HB[hb]-HB['Hips']))
    off=as_-hs; ol=(off*lat).sum(1); of_=(off*fwd).sum(1); ou=off[:,2]
    s_=1.0 if sd=='L' else -1.0
    ol2=ol-s_*FEM['narrow_m']/k          # feet land closer to the midline
    off2=ol2[:,None]*lat+(of_*FEM['stride'])[:,None]*fwd+ou[:,None]*zup
    T=Hj0+k*off2
    STATS[sd]=(Hj0,k*off,k*off2,lat,fwd)
    v=T-Hj; dd=np.linalg.norm(v,axis=1); u=v/dd[:,None]; dd=np.minimum(dd,(L1+L2)*0.9995); 
    pole=(ks-hs)-((ks-hs)*u).sum(1,keepdims=True)*u; pole=unit(pole)
    a=(L1**2-L2**2+dd**2)/(2*dd); h=np.sqrt(np.maximum(L1**2-a**2,1e-8))
    knee=Hj+a[:,None]*u+h[:,None]*pole
    return Hj,knee,Hj+dd[:,None]*u,T
LEGD={}; STATS={}
for side,sd in (('Left','L'),('Right','R')):
    Hj,knee,ank,T=leg_ik(side,sd); LEGD[sd]=(Hj,knee,ank)
def widths(off):   # lateral separation of the ankles about the hip midline, stance-weighted mean
    return None
hipwid=abs(rest['UpperLeg.L']['head'][0]-rest['UpperLeg.R']['head'][0])
swb=hipwid+((STATS['L'][1]-STATS['R'][1])*STATS['L'][3]).sum(1)  # lateral ankle separation relative to hips, BEFORE
swa=hipwid+((STATS['L'][2]-STATS['R'][2])*STATS['L'][3]).sum(1)
hipw=hipwid
print('ankle-to-ankle lateral distance mean m: before %.4f after %.4f (hip width %.3f)'%(swb.mean(),swa.mean(),hipw))
fb=np.abs((STATS['L'][1]-STATS['R'][1])*STATS['L'][4]).sum(1)
fa_=np.abs((STATS['L'][2]-STATS['R'][2])*STATS['L'][4]).sum(1)
print('fore-aft ankle separation p95 m (stride proxy): before %.3f after %.3f'%(np.percentile(fb,95),np.percentile(fa_,95)))
aim={'UpperArm.L':('LeftArm','LeftForeArm'),'LowerArm.L':('LeftForeArm','LeftHand'),'Hand.L':('LeftHand','LeftHandMiddleEnd'),
     'UpperArm.R':('RightArm','RightForeArm'),'LowerArm.R':('RightForeArm','RightHand'),'Hand.R':('RightHand','RightHandMiddleEnd'),
     'UpperLeg.L':('LeftLeg','LeftShin'),'LowerLeg.L':('LeftShin','LeftFoot'),'Foot.L':('LeftFoot','LeftToeBase'),
     'UpperLeg.R':('RightLeg','RightShin'),'LowerLeg.R':('RightShin','RightFoot'),'Foot.R':('RightFoot','RightToeBase')}
for b in BONES:
    if b in aim:
        p=PAR[b]; R0=np.einsum('fab,bc,cd->fad',Rw[p],RB[p].T,RB[b])
        d0=R0[:,:,1]; d1=unit(Pa[:,J(aim[b][1])]-Pa[:,J(aim[b][0])])
        if b.startswith('UpperArm') or b.startswith('LowerArm'):
            sa=1.0 if b.endswith('.L') else -1.0; ang_=np.tan(np.radians(FEM['upper_arm_in_deg'] if b.startswith('UpperArm') else FEM['forearm_in_deg']))
            d1=unit(d1-sa*ang_*lat)
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
    LW=[float(x) for x in os.environ.get('LOOP_WIN','0.9,1.6,7.8,%g'%((N-1)/30)).split(',')]
    for s_ in range(int(LW[0]*30),int(LW[1]*30)):
        for e_ in range(int(LW[2]*30),min(N-1,int(LW[3]*30)+1)):
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
