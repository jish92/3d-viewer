# retarget Kimodo (SOMA 77, y-up, +x her left, +z front, metres) output onto the swimher rig parameter dicts (ylib.Rig.solve)
import numpy as np, math, json
from ylib import *
NAMES=json.load(open('/workspace/me3d/swim_female/yoga/soma77_names.json'))
J=NAMES.index
def fr(up,lat):
    y=nrm(up); x=np.asarray(lat,float); x=nrm(x-y*(x@y)); z=np.cross(x,y); return np.stack([x,y,z],1)
def slerp_R(Ra,Rb,u):
    from scipy.spatial.transform import Rotation as R,Slerp
    return Slerp([0,1],R.from_matrix([Ra,Rb]))([u]).as_matrix()[0]
def smooth(a,k):
    a=np.asarray(a,float); pad=np.concatenate([np.repeat(a[:1],k,0),a,np.repeat(a[-1:],k,0)]); w=np.ones(2*k+1)/(2*k+1)
    return np.stack([np.convolve(pad[:,i],w,'valid') for i in range(a.shape[1])],1) if a.ndim==2 else np.convolve(pad,w,'valid')
def retarget(npz,rig,rest,prayer=None,loop_blend=10,cut=None,contact_thr=0.045,lock=True):
    d=np.load(npz); Jp=d['posed_joints'].astype(float)
    if cut: Jp=Jp[cut[0]:cut[1]]
    N=len(Jp)
    # standing reference = first frame
    p0=Jp[0]
    legK=np.linalg.norm(p0[J('LeftLeg')]-p0[J('LeftFoot')]); legR=np.linalg.norm(rig.head['UpperLeg.L']-rig.head['Foot.L'])
    k=legR/legK
    armK=np.linalg.norm(p0[J('LeftArm')]-p0[J('LeftForeArm')])+np.linalg.norm(p0[J('LeftForeArm')]-p0[J('LeftHand')]); armR=rig.L['UpperArm.L']+rig.L['LowerArm.L']
    ka=armR/armK
    pel0=rig.head['Hips']
    hips=Jp[:,J('Hips')]; hips_t=pel0+k*(hips-hips[0])
    # ankles with foot-plant
    A={}; 
    for sd,sn in (('L','Left'),('R','Right')):
        a=Jp[:,J(sn+'Foot')].copy(); A[sd]=a
    ank={}; plant={}
    for sd in 'LR':
        a=A[sd]; y0=a[:,1].min(); toe=Jp[:,J(('Left' if sd=='L' else 'Right')+'ToeBase')]
        low=(np.minimum(a[:,1],toe[:,1])<0.05+0.0)
        # smooth foot-speed test
        sp=np.r_[0,np.linalg.norm(np.diff(a[:,[0,2]],axis=0),axis=1)]*30
        con=low&(sp<0.25); 
        # clean: contact runs of >=4 frames
        ca=a.copy()
        i=0
        while i<N:
            if con[i]:
                j=i
                while j<N and con[j]: j+=1
                if lock and j-i>=4:
                    m=a[i:j,[0,2]].mean(0); ca[i:j,[0,2]]=m
                    # soften run edges: blend over 3 frames
                    for q in range(3):
                        w=(q+1)/4
                        if i-q-1>=0: ca[i-q-1,[0,2]]=a[i-q-1,[0,2]]*(1-w)+m*w
                        if j+q<N: ca[j+q,[0,2]]=a[j+q,[0,2]]*(1-w)+m*w
                i=j
            else: i+=1
        ank[sd]=rest['leg'+sd]['ankle']+k*(ca-a[0:1]); plant[sd]=con
    # torso frames
    Hf=[];Sf=[];Cf=[];Nf=[];Hdf=[]
    for t in range(N):
        p=Jp[t]
        lat_h=p[J('LeftLeg')]-p[J('RightLeg')]
        Hf.append(fr(p[J('Spine2')]-p[J('Hips')],lat_h))
        Cf.append(fr(p[J('Neck1')]-p[J('Spine2')]+0.0*(p[J('Neck1')]-p[J('Hips')]),p[J('LeftArm')]-p[J('RightArm')]))
        Nf.append(fr(p[J('Head')]-p[J('Neck1')],p[J('LeftArm')]-p[J('RightArm')]))
        Hdf.append(fr(p[J('HeadEnd')]-p[J('Head')],p[J('LeftEye')]-p[J('RightEye')]))
    Hf=np.array(Hf);Cf=np.array(Cf);Nf=np.array(Nf);Hdf=np.array(Hdf)
    # reference frame 0 (standing) -> rig rest orientation = identity; remove the standing offset so frame 0 is rig neutral
    off=lambda F: np.einsum('fab,bc->fac',F,F[0].T)
    Hd,Cd,Nd,Hdd=off(Hf),off(Cf),off(Nf),off(Hdf)
    f0=lambda v:nrm(v)
    pitch0={}; yaw0={}
    Ps=[]
    for t in range(N):
        p=Jp[t]; P={}
        P['pelvis']=hips_t[t].copy(); P['Rhips']=Hd[t]
        P['Rspine']=slerp_R(Hd[t],Cd[t],0.5); P['Rchest']=Cd[t]; P['Rneck']=slerp_R(Cd[t],Nd[t],0.5)@np.eye(3); P['Rhead']=Hdd[t]
        # neck frame: chest->neck handled; Rneck = neck world delta
        P['Rneck']=Nd[t]
        for sd,sn in (('L','Left'),('R','Right')):
            toe=p[J(sn+'ToeBase')]; fo=p[J(sn+'Foot')]; v=toe-fo; v0=Jp[0][J(sn+'ToeBase')]-Jp[0][J(sn+'Foot')]
            ya=math.degrees(math.atan2(v[0],v[2])); ya0=math.degrees(math.atan2(v0[0],v0[2]))
            pi_=math.degrees(math.asin(np.clip(-v[1]/np.linalg.norm(v),-1,1))); pi0=math.degrees(math.asin(np.clip(-v0[1]/np.linalg.norm(v0),-1,1)))
            kn=p[J(sn+'Shin')]; hp=p[J(sn+'Leg')]
            pole=nrm((kn-hp)-((kn-hp)@nrm(fo-hp))*nrm(fo-hp)) if np.linalg.norm((kn-hp)-((kn-hp)@nrm(fo-hp))*nrm(fo-hp))>1e-4 else np.array([0,0,1.])
            # base rig rest: ankle y given by contact; yaw/pitch relative to standing frame
            P['leg'+sd]=dict(ankle=ank[sd][t].copy(),pole=pole,yaw=ya-ya0,pitch=pi_-pi0,roll=0)
            # arms (chest mode)
            sh=p[J(sn+'Arm')]; el=p[J(sn+'ForeArm')]; wr=p[J(sn+'Hand')]; mid=p[J(sn+'HandMiddleEnd')]
            rel=ka*(Cf[t].T@(wr-sh))           # in chest frame of the source
            # rig wrist in chest-local = rig shoulder offset from chest + rel (chest-local coords use the rig chest delta = Cd; convert source-chest-frame vector -> world delta frame)
            relw=Cf[t]@ (Cf[0].T@Cf[0]) @ np.zeros(3)  # unused
            wr_local=(rig.head['UpperArm.'+sd]-rig.head['Chest'])+ (Cf[0]@(Cf[t].T@(wr-sh)))*ka
            # elbow pole (world)
            ln=nrm(wr-sh); pl=(el-sh)-((el-sh)@ln)*ln; pl=nrm(pl) if np.linalg.norm(pl)>1e-4 else np.array([0,-0.3,-1.])
            hd=nrm(mid-wr); sgn=1 if sd=='L' else -1
            P['arm'+sd]=dict(mode='chest',wrist=wr_local,pole=pl,hand=hd,palm=np.array([-sgn,0,0.]))
        P['_hips_y']=hips[t,1]
        Ps.append(P)
    return Ps,dict(k=k,ka=ka,N=N,plant=plant)
