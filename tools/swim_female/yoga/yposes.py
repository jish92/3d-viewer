import numpy as np, math
from ylib import *
d2r=math.radians
def dirxz(yaw,y=0.0): return nrm([math.sin(d2r(yaw)),y,math.cos(d2r(yaw))])
def copyP(P): return {k:(dict(v) if isinstance(v,dict) else v) for k,v in P.items()}
LMAX=0.819; ANK_Y=0.097
def hipY(P_ankle,x_h,pz=0.0):
    pass
def stand_py(rest,foot_x,fz=None,k=0.992):
    """pelvis height for straight (k*max) legs with feet at +-foot_x"""
    a=rest['legL']['ankle']; dx=foot_x-0.106; dz=(a[2] if fz is None else fz)-0.003
    return a[1]+math.sqrt((k*LMAX)**2-dx*dx-dz*dz)+0.012
def rest_chest_arms(rig,rest):
    P=copyP(rest); ch=rig.head['Chest']
    for sd in 'LR':
        a=dict(rest['arm'+sd]); a['mode']='chest'; a['wrist']=a['wrist']-ch; P['arm'+sd]=a
    return P
def base_stance(rig,rest,foot_x=0.21,toe=32,knee_yaw=40):
    P=rest_chest_arms(rig,rest)
    P['pelvis']=np.array([0,stand_py(rest,foot_x),rig.head['Hips'][2]])
    for sd,s in (('L',1),('R',-1)):
        a=rest['leg'+sd]['ankle'].copy(); a[0]=s*foot_x
        P['leg'+sd]=dict(ankle=a,pole=dirxz(s*knee_yaw,0.0),yaw=s*toe,pitch=0,roll=0)
    return P
# ---------------------------------------------------------------- 1 deep squat (malasana)
def squat(rig,rest,py=0.26,pz=-0.05,foot_x=0.21,toe=32,knee_yaw=42,hip=(0,30,0),spine=(0,16,0),chest=(0,8,0),neck=(0,4,0),head=(0,0,0)):
    P=base_stance(rig,rest,foot_x,toe,knee_yaw); P['pelvis']=np.array([0,py,pz]); P['hips']=hip; P['spine']=spine; P['chest']=chest; P['neck']=neck; P['head']=head
    for sd,s in (('L',1),('R',-1)):
        P['leg'+sd]['pole']=dirxz(s*knee_yaw,0.1)
        P['arm'+sd]=dict(mode='chest',wrist=np.array([s*0.022,-0.03,0.20]),pole=nrm([s*0.7,-0.1,-0.5]),hand=nrm([0,0.8,0.6]),palm=np.array([-s,0,0.]))
    return P
def squat_stand(rig,rest,**kw): return base_stance(rig,rest,kw.get('foot_x',0.21),kw.get('toe',32),kw.get('knee_yaw',42))
# ---------------------------------------------------------------- 2 warrior II
def warrior(rig,rest,px=-0.12,py=0.60,front_x=0.43,back_k=0.992,toe_back=22,head_yaw=75,neck_yaw=30):
    P=base_stance(rig,rest,0.21,0,0); P['pelvis']=np.array([px,py,rig.head['Hips'][2]]); 
    hy=py-0.012; hipL=np.array([px+0.106,hy,0.003]); hipR=np.array([px-0.106,hy,0.003])
    aL=np.array([front_x,ANK_Y,0.026]); P['legL']=dict(ankle=aL,pole=nrm([1,0.5,0.05]),yaw=90,pitch=0,roll=0)
    dy=hy-ANK_Y; dx=math.sqrt((back_k*LMAX)**2-dy*dy)
    aR=np.array([hipR[0]-dx,ANK_Y,0.026]); P['legR']=dict(ankle=aR,pole=nrm([-0.2,0.5,1.0]),yaw=toe_back,pitch=0,roll=0)
    P['hips']=(0,0,0); P['neck']=(neck_yaw,0,0); P['head']=(head_yaw,0,0)
    for sd,s in (('L',1),('R',-1)):
        P['arm'+sd]=dict(mode='chest',wrist=np.array([s*0.648,0.158,0.03]),pole=nrm([0,-0.3,-1]),hand=np.array([s,0,0.]),palm=np.array([0,-1,0.]))
    return P
def warrior_stand(rig,rest,**kw):
    P=warrior(rig,rest,**kw); S=base_stance(rig,rest,0.21,0,0)
    # same feet as the pose, standing tall (feet pivot only during the stand)
    S['legL']=dict(P['legL']); S['legR']=dict(P['legR'])
    return S
# ---------------------------------------------------------------- 3 tree
def tree(rig,rest,px=0.045,stand_x=0.10,ankle=(-0.012,0.80,0.0),knee_out=-1.0,roll=80,pitch=35,arms='up'):
    P=rest_chest_arms(rig,rest)
    S=stand_x
    aL=rest['legL']['ankle'].copy(); aL[0]=S
    hy=stand_py(rest,0.10)  # pelvis height for standing leg
    P['pelvis']=np.array([px,hy-0.002,rig.head['Hips'][2]]); P['hips']=(0,0,-3)
    P['legL']=dict(ankle=aL,pole=nrm([0.1,0,1]),yaw=3,pitch=0,roll=0)
    a=np.array(ankle,float)
    P['legR']=dict(ankle=a,pole=nrm([knee_out,0.15,0.35]),yaw=0,pitch=pitch,roll=roll)
    for sd,s in (('L',1),('R',-1)):
        if arms=='up': P['arm'+sd]=dict(mode='chest',wrist=np.array([s*0.045,0.158+0.46,0.0]),pole=nrm([s*1,0.0,0.25]),hand=np.array([0,1,0.]),palm=np.array([-s,0,0.]))
        else: P['arm'+sd]=dict(mode='chest',wrist=np.array([s*0.022,-0.03,0.20]),pole=nrm([s*0.7,-0.1,-0.5]),hand=nrm([0,0.8,0.6]),palm=np.array([-s,0,0.]))
    return P
def tree_stand(rig,rest,**kw):
    P=tree(rig,rest,**kw); S=rest_chest_arms(rig,rest); 
    S['pelvis']=P['pelvis'].copy(); S['hips']=P['hips']; S['legL']=dict(P['legL'])
    aR=rest['legR']['ankle'].copy(); aR[0]=-0.10; S['legR']=dict(ankle=aR,pole=nrm([0,0,1]),yaw=-3,pitch=0,roll=0)
    return S

# ---------------------------------------------------------------- world-mode arms helper
def arm_world(rig,P,rel_L=None,rel_R=None,**kw):
    """convert chest-relative wrist targets to world targets for the torso pose in P (arms of P are ignored)"""
    P2=copyP(P)
    for sd in 'LR':
        if P2['arm'+sd].get('mode')!='chest': P2['arm'+sd]=dict(P2['arm'+sd],mode='chest',wrist=np.zeros(3))
    D,pos,info=rig.solve(P2)
    return pos['Chest'],D['Chest']
def to_world(rig,P,rel,sd,pole,hand,palm):
    pc,Dc=arm_world(rig,P)
    return dict(mode='world',wrist=pc+Dc@np.asarray(rel,float),pole=nrm(pole),hand=nrm(hand),palm=np.asarray(palm,float))
def arms_down_world(rig,P):
    for sd,s in (('L',1),('R',-1)):
        P['arm'+sd]=to_world(rig,P,[s*0.205,-0.20,0.04],sd,[s*0.3,-0.3,-1],[0,-1,0.2],[0,0,-s*1.0])
    return P
# ---------------------------------------------------------------- 4 downward dog
DBG=False
def torso_pitch(th,hips=0.82,spine=0.93,neck_add=20,head_add=32):
    return dict(hips=(0,th*hips,0),spine=(0,th*spine,0),chest=(0,th,0),neck=(0,th+neck_add,0),head=(0,th+head_add,0))
def dog(rig,rest,hy=0.68,ay=0.157,heel_pitch=20,wy=0.072,hand_x=0.17,foot_x=0.11,th=None,toe=0,arm_k=0.985,line=-3):
    P=base_stance(rig,rest,foot_x,toe,0)
    dy=hy-ay; dz=math.sqrt((0.992*LMAX)**2-dy*dy); pz=0.026+dz+0.003
    py=hy+0.012; P['pelvis']=np.array([0,py,pz])
    for sd,s in (('L',1),('R',-1)):
        P['leg'+sd]=dict(ankle=np.array([s*foot_x,ay,0.026]),pole=nrm([0,0.1,1]),yaw=s*toe,pitch=heel_pitch,roll=0)
    Lm=(rig.L['UpperArm.L']+rig.L['LowerArm.L'])*arm_k
    def build(t):
        Q=copyP(P); Q.update(torso_pitch(t))
        D,pos,info=rig.solve(Q); sh=pos['UpperArm.L']
        dyy=sh[1]-wy; dzz=math.sqrt(max(Lm**2-dyy*dyy,1e-6)); zw=sh[2]+dzz
        for sd,s in (('L',1),('R',-1)):
            Q['arm'+sd]=dict(mode='world',wrist=np.array([s*hand_x,wy,zw]),pole=nrm([s,0.3,0]),hand=np.array([0,0,1.]),palm=np.array([0,-1,0.]))
        arm_ang=math.degrees(math.atan2(dyy,dzz)); return Q,arm_ang
    if th is not None: return build(th)[0]
    best=None
    for t in np.arange(100,171,1.0):
        Q,aa=build(t); err=abs(aa-(t-90)-line)
        if DBG: print('th',t,round(aa,1),round(err,1))
        if best is None or err<best[0]: best=(err,Q,t)
    if DBG: print('chosen',best[2])
    return best[1]
# ---------------------------------------------------------------- kneeling family (puppy / child)
KAY=0.055; KPITCH=168     # ankle height with the dorsum on the floor (calibrated)
def kneel_legs(P,pz,foot_x=0.11,ay=None,knee_x=None):
    ay=KAY if ay is None else ay
    hy=P['pelvis'][1]-0.012
    for sd,s in (('L',1),('R',-1)):
        kx=s*foot_x
        P['leg'+sd]=dict(ankle=np.array([kx,ay,pz-0.408]),pole=nrm([0,-0.2,1]),yaw=0,pitch=KPITCH,roll=0)
    return P
def kneel_upright(rig,rest,pz=-0.30,foot_x=0.11,ay=None,py=0.475):
    P=base_stance(rig,rest,foot_x,0,0); P['pelvis']=np.array([0,py,pz]); return kneel_legs(P,pz,foot_x,ay)
def puppy(rig,rest,pz=-0.30,py=0.475,hand_x=0.20,wy=0.076,zw_off=0.80,th=None,foot_x=0.11,hips=62,neck_add=22,head_add=40,arm_k=0.985):
    P=kneel_upright(rig,rest,pz,foot_x,None,py)
    Lm=(rig.L['UpperArm.L']+rig.L['LowerArm.L'])*arm_k
    def build(t):
        Q=copyP(P); Q.update(torso_pitch(t,hips/t if t else 1,0.92,neck_add,head_add))
        Q['hips']=(0,hips,0)
        D,pos,info=rig.solve(Q); sh=pos['UpperArm.L']; dyy=sh[1]-wy; dzz=math.sqrt(max(Lm**2-dyy*dyy,1e-6)); 
        zw=sh[2]+dzz
        for sd,s in (('L',1),('R',-1)):
            Q['arm'+sd]=dict(mode='world',wrist=np.array([s*hand_x,wy,zw]),pole=nrm([s,0.4,0]),hand=np.array([0,0,1.]),palm=np.array([0,-1,0.]))
        return Q
    return build(th if th is not None else 112)

def child(rig,rest,pz=-0.30,back=0.10,py=0.30,kpole=0.3,hand_x=0.19,wy=0.076,th=None,foot_x=0.11,hips=84,neck_add=18,head_add=22,arm_k=0.985,spine_k=0.97):
    P=kneel_upright(rig,rest,pz,foot_x,None,0.475)
    P['pelvis']=np.array([0,py,pz-0.408+back])
    for sd in 'LR': P['leg'+sd]['pole']=nrm([0,kpole,1])
    Lm=(rig.L['UpperArm.L']+rig.L['LowerArm.L'])*arm_k
    t=th if th is not None else 94
    Q=copyP(P); Q.update(torso_pitch(t,hips/t,spine_k,neck_add,head_add)); Q['hips']=(0,hips,0)
    D,pos,info=rig.solve(Q); sh=pos['UpperArm.L']; dyy=sh[1]-wy; dzz=math.sqrt(max(Lm**2-dyy*dyy,1e-6)); zw=sh[2]+dzz
    for sd,s in (('L',1),('R',-1)):
        Q['arm'+sd]=dict(mode='world',wrist=np.array([s*hand_x,wy,zw]),pole=nrm([s,0.4,0]),hand=np.array([0,0,1.]),palm=np.array([0,-1,0.]))
    return Q

# ---------------------------------------------------------------- happy baby (lying on back)
def lying_base(rig,rest,P,py,pz,th=-90,hips=0.93,neck_add=12,head_add=10):
    Q=copyP(P); Q['pelvis']=np.array([0,py,pz]); Q.update(torso_pitch(th,hips,0.98,neck_add,head_add)); return Q
def hb_legs(P,ank,pole,pitch=168,foot_x=0.0):
    for sd,s in (('L',1),('R',-1)):
        P['leg'+sd]=dict(ankle=np.array([s*ank[0],ank[1],ank[2]]),pole=nrm([s*pole[0],pole[1],pole[2]]),yaw=0,pitch=pitch,roll=s*8)
    return P
def hb_arms_world(rig,Q,wr,pole,palm=(0,-1,0.)):
    for sd,s in (('L',1),('R',-1)):
        Q['arm'+sd]=dict(mode='world',wrist=np.array([s*wr[0],wr[1],wr[2]]),pole=nrm([s*pole[0],pole[1],pole[2]]),hand=np.array([0,0,1.]),palm=np.array(palm))
    return Q
def happybaby(rig,rest,py=0.13,pz=-0.25,ank=(0.28,0.62,-0.47),pole=(0.9,0.5,-0.7),wr=(0.31,0.62,-0.47),apole=(1,-0.3,0.2),pitch=168):
    S=base_stance(rig,rest,0.11,0,0)
    Q=lying_base(rig,rest,S,py,pz)
    hb_legs(Q,ank,pole,pitch)
    hb_arms_world(rig,Q,wr,apole,palm=(0,0,-1.))
    return Q
def hb_sit(rig,rest,py=0.10,pz=-0.25):
    S=base_stance(rig,rest,0.11,0,0)
    Q=lying_base(rig,rest,S,py,pz,th=-20,hips=0.9,neck_add=15,head_add=10)
    hb_legs(Q,(0.18,0.097,0.0),(0.2,1,0.5),pitch=0)
    hb_arms_world(rig,Q,(0.20,0.04,-0.55),(1,0.2,-0.5),palm=(0,1,0.))
    return Q
def hb_lie(rig,rest,py=0.11,pz=-0.25):
    Q=hb_sit(rig,rest,py,pz); S=base_stance(rig,rest,0.11,0,0)
    Q.update(torso_pitch(-90,0.93,0.98,12,10))
    hb_arms_world(rig,Q,(0.26,0.05,-0.95),(1,0.3,0),palm=(0,1,0.))
    return Q
