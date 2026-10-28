import numpy as np, math, copy
from ylib import *
import yposes as yp
from yclips import *
def clip_squat(rig,rest):
    S0=yp.squat_stand(rig,rest); S1=yp.squat(rig,rest)
    keys,h=there_and_back([(0,S0,None),(0.6,S0,None),(2.7,S1,None)],2.4)
    return Clip('Yoga_DeepSquat',keys,h)
def clip_tree(rig,rest):
    S0=yp.tree_stand(rig,rest); S1=yp.tree(rig,rest)
    keys,h=there_and_back([(0,S0,None),(0.6,S0,None),(2.8,S1,{'legR':0.0})],2.4)
    return Clip('Yoga_Tree',keys,h,breath=0.8)
def clip_warrior(rig,rest,**kw):
    S0=yp.base_stance(rig,rest,0.12,0,0); W2=yp.warrior(rig,rest,**kw)
    W1=copy.deepcopy(W2); W1['legL']=copy.deepcopy(S0['legL']); W1['legL']['pole']=nrm([0,0,1]); W1['legL']['yaw']=0
    for k in ('arml','armL','armR','neck','head'): 
        if k in S0: W1[k]=S0[k]
    W1['neck']=(0,0,0); W1['head']=(0,0,0)
    keys,h=there_and_back([(0,S0,None),(0.6,S0,None),(1.9,W1,{'legR':0.14}),(2.0,W2,{'legL':0.14})],2.4)
    return Clip('Yoga_WarriorII',keys,h)
def clip_dog(rig,rest,**kw):
    S1=yp.dog(rig,rest,**kw)
    S0=yp.base_stance(rig,rest,0.11,0,0); S0=yp.arms_down_world(rig,S0)
    # forward fold: hands planted near the feet, hips high
    M=copy.deepcopy(S1); M['pelvis']=np.array([0,0.84,0.30]); M.update(yp.torso_pitch(168,0.80,0.93,12,18))
    for sd in 'LR': M['leg'+sd]=dict(S0['leg'+sd]); M['leg'+sd]['pitch']=0
    for sd,s in (('L',1),('R',-1)):
        M['arm'+sd]=dict(S1['arm'+sd]); M['arm'+sd]['wrist']=np.array([s*0.17,0.076,0.52]); M['arm'+sd]['hand']=np.array([0,0,1.])
    keys,h=there_and_back([(0,S0,None),(0.6,S0,None),(2.2,M,None),(2.2,S1,{'armL':0.07,'armR':0.07})],2.4)
    return Clip('Yoga_DownwardDog',keys,h,breath=0.9)
def clip_puppy(rig,rest,**kw):
    S1=yp.puppy(rig,rest,th=124,neck_add=18,head_add=18,hips=100,wy=0.076,**kw)
    S0=yp.base_stance(rig,rest,0.11,0,0);     S0=yp.arms_down_world(rig,S0)
    pz=-0.30
    # K1 lunge: right foot stepped back (toes planted), left foot planted
    K1=copy.deepcopy(S0); K1['pelvis']=np.array([0,0.62,pz]); K1['hips']=(0,8,0); K1['spine']=(0,10,0); K1['chest']=(0,6,0)
    K1['legR']=dict(ankle=np.array([-0.11,0.14,pz-0.408]),pole=nrm([0,-0.1,1]),yaw=0,pitch=60,roll=0)
    K1['legL']=dict(S0['legL']); K1['legL']['pole']=nrm([0,0.2,1])
    K1=yp.arms_down_world(rig,K1)
    # K2 right knee down, left foot still planted
    K2=copy.deepcopy(K1); K2['pelvis']=np.array([0,0.475,pz]); K2['legR']=dict(S1['legR']); K2['legL']['pole']=nrm([0,0.3,1])
    K2=yp.arms_down_world(rig,K2)
    # K3 both knees down, upright
    K3=copy.deepcopy(K2); K3['legL']=dict(S1['legL']); K3=yp.arms_down_world(rig,K3)
    keys,h=there_and_back([(0,S0,None),(0.6,S0,None),(1.8,K1,{'legR':0.14}),(1.5,K2,None),(1.5,K3,{'legL':0.14}),(2.4,S1,{'armL':0.06,'armR':0.06})],2.6)
    return Clip('Yoga_PuppyPose',keys,h,breath=0.8)

def clip_child(rig,rest,**kw):
    S1=yp.child(rig,rest,**kw)
    S0=yp.base_stance(rig,rest,0.11,0,0);     S0=yp.arms_down_world(rig,S0)
    pz=-0.30
    K1=copy.deepcopy(S0); K1['pelvis']=np.array([0,0.62,pz]); K1['hips']=(0,8,0); K1['spine']=(0,10,0); K1['chest']=(0,6,0)
    K1['legR']=dict(ankle=np.array([-0.11,0.14,pz-0.408]),pole=nrm([0,-0.1,1]),yaw=0,pitch=60,roll=0)
    K1['legL']=dict(S0['legL']); K1['legL']['pole']=nrm([0,0.2,1])
    K1=yp.arms_down_world(rig,K1)
    K2=copy.deepcopy(K1); K2['pelvis']=np.array([0,0.475,pz]); K2['legR']=dict(yp.kneel_upright(rig,rest,pz)['legR']); K2['legL']['pole']=nrm([0,0.3,1])
    K2=yp.arms_down_world(rig,K2)
    K3=yp.kneel_upright(rig,rest,pz); K3=yp.arms_down_world(rig,K3)
    for sd in 'LR': K3['leg'+sd]['pole']=nrm([0,0.3,1])
    K3b=copy.deepcopy(K3); K3b['pelvis']=np.array([0,0.40,-0.62]); K3b['hips']=(0,40,0); K3b['spine']=(0,45,0); K3b['chest']=(0,50,0)
    keys,h=there_and_back([(0,S0,None),(0.6,S0,None),(1.8,K1,{'legR':0.14}),(1.5,K2,None),(1.5,K3,{'legL':0.14}),(0.9,K3b,None),(1.7,S1,{'armL':0.06,'armR':0.06})],2.6)
    return Clip('Yoga_Child',keys,h,breath=0.7)

def clip_happybaby(rig,rest,**kw):
    S1=yp.happybaby(rig,rest,**kw)
    S0=yp.base_stance(rig,rest,0.11,0,0);     S0=yp.arms_down_world(rig,S0)
    Qs=yp.squat(rig,rest,py=0.30,pz=-0.10,foot_x=0.18,toe=10,knee_yaw=10,hip=(0,70,0),spine=(0,30,0),chest=(0,15,0))
    Qs=yp.arms_down_world(rig,Qs)
    K1=yp.hb_sit(rig,rest); K2=yp.hb_lie(rig,rest)
    keys,h=there_and_back([(0,S0,None),(0.6,S0,None),(1.8,Qs,None),(1.4,K1,None),(1.4,K2,None),(2.2,S1,None)],2.6)
    return Clip('Yoga_HappyBaby',keys,h,breath=0.7)
