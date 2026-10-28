import numpy as np,json,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N=json.load(open('/workspace/me3d/swim_female/yoga/soma77_names.json')); J=N.index
B=[('Hips','Spine1'),('Spine1','Spine2'),('Spine2','Chest'),('Chest','Neck1'),('Neck1','Head'),('Chest','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),('Chest','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),
('Hips','LeftLeg'),('LeftLeg','LeftShin'),('LeftShin','LeftFoot'),('LeftFoot','LeftToeBase'),('Hips','RightLeg'),('RightLeg','RightShin'),('RightShin','RightFoot'),('RightFoot','RightToeBase')]
name=sys.argv[1]; P=np.load('/workspace/me3d/kim_out/%s.npz'%name)['posed_joints']; T=len(P); fr=np.linspace(30,T-60,4).astype(int)
fig,ax=plt.subplots(2,4,figsize=(12,7))
for i,f in enumerate(fr):
    for r,(a,b,lab) in enumerate(((0,1,'front'),(2,1,'side'))):
        A=ax[r,i]
        for u,v in B:
            p,q=P[f][J(u)],P[f][J(v)]; A.plot([p[a],q[a]],[p[b],q[b]],'k-')
        A.set_xlim(-0.95,0.95); A.set_ylim(-0.05,1.9); A.set_aspect('equal',adjustable='box'); A.set_title('%d %s'%(f,lab),fontsize=8); A.axis('off')
plt.tight_layout(); plt.savefig('/tmp/sh5_%s.png'%name,dpi=80)
