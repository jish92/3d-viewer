import bpy, sys, math, os, json, numpy as np
from mathutils import Vector, Matrix, Quaternion
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.interpolate import PchipInterpolator
OUT=sys.argv[sys.argv.index('--')+1]
LIFT=0.0245
JJ={k:np.array(v)+np.array([0,LIFT,0]) for k,v in json.load(open('/workspace/me3d/swim_female/joints_y.json')).items()}
# ---- female gait parameters (documented in README) ----
FEM=json.load(open('/workspace/me3d/swim_female/gait_params.json'))
bpy.ops.wm.read_factory_settings(use_empty=True)
D=np.load('/workspace/me3d/swim_female/her_parts.npz',allow_pickle=True)
Vg=D['V'];Fg=D['F'];Cg=D['C'];part=D['part']
def g2b(p): return Vector((p[0],-p[2],p[1]))
# ---------------- skeleton ----------------
HB=JJ['l-upper-leg'][1]; HZ0=JJ['l-upper-leg'][2]
P=lambda n,dz=None: tuple(JJ[n])
hips=(0,JJ['pelvis'][1],0.02); spine=(0,JJ['spine-3'][1],0.005); chest=(0,JJ['spine-2'][1]+0.07,0.0); neck=(0,JJ['neck'][1],0.0); head=(0,JJ['head'][1],0.04); htop=(0,JJ['head-2'][1],0.034)
bones=[('Hips',None,hips,spine,0.16),('Spine','Hips',spine,chest,0.15),('Chest','Spine',chest,neck,0.17),('Neck','Chest',neck,head,0.05),('Head','Neck',head,htop,0.10)]
for sd,s in (('L',1),('R',-1)):
    m=np.array([s,1,1.])
    sh=JJ['l-shoulder']*m; el=JJ['l-elbow']*m; wr=JJ['l-hand']*m; ft=JJ['l-finger-3-4']*m
    hip=JJ['l-upper-leg']*m; kn=JJ['l-knee']*m; an=JJ['l-ankle']*m; toe=np.array([0.18*s,0.0475,0.205])
    bones+=[('UpperArm.'+sd,'Chest',sh,el,0.055),('LowerArm.'+sd,'UpperArm.'+sd,el,wr,0.045),('Hand.'+sd,'LowerArm.'+sd,wr,ft,0.035),
            ('UpperLeg.'+sd,'Hips',hip,kn,0.10),('LowerLeg.'+sd,'UpperLeg.'+sd,kn,an,0.065),('Foot.'+sd,'LowerLeg.'+sd,an,toe,0.07)]
# bust helper bones (children of Chest)
BC=np.array([0.095,1.275+LIFT,0.130])
for sd,s in (('L',1),('R',-1)):
    h=BC*np.array([s,1,1]); bones.append(('Bust.'+sd,'Chest',tuple(h),tuple(h+np.array([0,0.05,0])),0.06))
# glute helper bones (children of Hips)
GC=np.array([0.085,0.89+LIFT,-0.095])
for sd,s_ in (('L',1),('R',-1)):
    h=GC*np.array([s_,1,1]); bones.append(('Glute.'+sd,'Hips',tuple(h),tuple(h+np.array([0,0.05,0])),0.06))
arm_d=bpy.data.armatures.new('Armature'); arm=bpy.data.objects.new('Armature',arm_d); bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active=arm; arm.select_set(True); bpy.ops.object.mode_set(mode='EDIT')
for n,p,h,t,r in bones:
    b=arm_d.edit_bones.new(n); b.head=g2b(h); b.tail=g2b(t)
    if p: b.parent=arm_d.edit_bones[p]; b.use_connect=False
bpy.ops.object.mode_set(mode='OBJECT')
# ---------------- weights ----------------
def seg_dist(P,a,b):
    ab=b-a; t=np.clip(((P-a)@ab)/(ab@ab),0,1); return np.linalg.norm(P-(a+t[:,None]*ab),axis=1)
names=[b[0] for b in bones]; nb=len(bones)
core=[b for b in bones if not b[0].startswith(('Bust','Glute'))]
isskin=(part=='skin')
Vs=Vg[isskin]
segs=[]
for n,p,h,t,r in core:
    a=np.array(h,float); b_=np.array(t,float)
    if n=='Hips': a=np.array([0,0.82+LIFT,0.02])
    segs.append((a,b_,r))
D_=np.stack([np.maximum(seg_dist(Vs,a,b_)-r,0) for a,b_,r in segs],1)
W=1.0/(D_+0.012)**3
idx=np.argsort(-W,1)[:,:4]; M=np.zeros_like(W); np.put_along_axis(M,idx,np.take_along_axis(W,idx,1),1); W=M/M.sum(1,keepdims=True)
# skin adjacency from the full body face list
db=np.load('/workspace/me3d/swim_female/stage_body.npz'); Fb=db['F']; Nb=len(db['Vb'])
E=np.concatenate([Fb[:,[0,1]],Fb[:,[1,2]],Fb[:,[2,0]]]); A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(Nb,Nb)); A=(A+A.T).tocsr(); A.data[:]=1; deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
HEADT=1.515
cn=[b[0] for b in core]
hi=cn.index('Head')
for _ in range(4):
    Wn=(A@W)/deg[:,None]; W=0.5*W+0.5*Wn
    W[Vs[:,1]>HEADT]=0; W[Vs[:,1]>HEADT,hi]=1
idx=np.argsort(-W,1)[:,:4]; M=np.zeros_like(W); np.put_along_axis(M,idx,np.take_along_axis(W,idx,1),1); W=M/np.maximum(M.sum(1,keepdims=True),1e-9)
# hip-flexion fix: upper-front thigh flesh stays partly with the pelvis (flesh compresses in the crease instead of the thigh cutting through the suit)
FRONT_HIP=float(os.environ.get('FRONT_HIP','0.9'))
if FRONT_HIP>0:
    ih=cn.index('Hips'); 
    for sd,s_ in (('L',1),('R',-1)):
        il=cn.index('UpperLeg.'+sd)
        fy=np.clip((Vs[:,1]-0.70)/0.14,0,1); fy=fy*fy*(3-2*fy)
        fz=np.clip((Vs[:,2]-0.02)/0.06,0,1)
        fx=np.clip(Vs[:,0]*s_/0.04,0,1)*np.clip((0.26-Vs[:,0]*s_)/0.05,0,1)
        f=FRONT_HIP*fy*fz*fx
        mv=W[:,il]*f; W[:,il]-=mv; W[:,ih]+=mv
# transfer to all vertices
Wall=np.zeros((len(Vg),len(cn)))
Wall[isskin]=W
tr=cKDTree(Vs)
# suit: weights of the nearest underlying skin vertex ; head parts rigid; slides on the foot bones
for pn in ('suit','strap','sole'):
    m=(part==pn)
    if pn=='suit':
        # use underlying body positions
        sv=db['Vbc']+np.array([0,LIFT,0]); _,ii=tr.query(sv); 
        ids=np.where(m)[0]; Wsu=W[ii]
        K=float(os.environ.get('SUIT_SHARP','2.5'))
        if K!=1:   # sharpen the hips/leg blend near the leg openings so the hem follows the thigh instead of being swallowed by it
            low=(Vg[ids,1]<0.98)[:,None]
            Wk=Wsu**K; Wk/=np.maximum(Wk.sum(1,keepdims=True),1e-9); Wsu=np.where(low,Wk,Wsu)
        Wall[ids]=Wsu
    else:
        _,ii=tr.query(Vg[m]); Wall[m]=W[ii]
for pn in ('hair','bun','eye','brow'):
    m=(part==pn); Wall[m]=0; Wall[m,hi]=1
# soles: fully on the foot bone of their side
for s_,sd in ((1,'L'),(-1,'R')):
    m=(part=='sole')&(Vg[:,0]*s_>0); Wall[m]=0; Wall[m,cn.index('Foot.'+sd)]=1
    m=(part=='strap')&(Vg[:,0]*s_>0); Wall[m]=0; Wall[m,cn.index('Foot.'+sd)]=1
# skin of the foot below the ankle: rigid on Foot (keeps toes + slide in sync)
for s_,sd in ((1,'L'),(-1,'R')):
    m=(part=='skin')&(Vg[:,0]*s_>0.05)&(Vg[:,1]<0.075)
    Wall[m]=Wall[m]*0.0; Wall[m,cn.index('Foot.'+sd)]=1
# full matrix incl. bust bones
Wf=np.zeros((len(Vg),nb)); 
for j,n in enumerate(cn): Wf[:,names.index(n)]=Wall[:,j]
# bust weights: skin+suit near the bust centres, front half
for sd,s in (('L',1),('R',-1)):
    c=BC*np.array([s,1,1])
    d=np.linalg.norm((Vg-c)*np.array([1,1.0,1.0]),axis=1)
    t=np.clip((0.115-d)/(0.115-0.035),0,1); wb=t*t*(3-2*t)
    fr=np.clip((Vg[:,2]-0.035)/0.04,0,1)
    tw=Wf[:,names.index('Chest')]+Wf[:,names.index('Spine')]
    wb=wb*fr*np.isin(part,['skin','suit'])*np.clip((tw-0.35)/0.4,0,1)
    wb=np.minimum(wb,0.98)
    j=names.index('Bust.'+sd)
    Wf*=(1-wb)[:,None]; Wf[:,j]=wb
# glute weights: skin+suit around the buttock centres, back half
for sd,s_ in (('L',1),('R',-1)):
    c=GC*np.array([s_,1,1])
    d=np.linalg.norm(Vg-c,axis=1)
    t=np.clip((0.115-d)/(0.115-0.03),0,1); wg=t*t*(3-2*t)
    bk=np.clip((-Vg[:,2]-0.02)/0.04,0,1)
    tw=Wf[:,names.index('Hips')]+0.5*Wf[:,names.index('UpperLeg.'+sd)]
    wg=wg*bk*np.isin(part,['skin','suit'])*np.clip((tw-0.3)/0.4,0,1)*(Vg[:,0]*s_>-0.01)
    wg=np.minimum(wg,0.95)
    j=names.index('Glute.'+sd)
    Wf*=(1-wg)[:,None]; Wf[:,j]=wg
Wf/=np.maximum(Wf.sum(1,keepdims=True),1e-9)
SIMP=os.environ.get('SIMP')
if SIMP:
    import trimesh
    tm=trimesh.load(SIMP,force='mesh',process=False); V2=np.asarray(tm.vertices); F2=np.asarray(tm.faces)
    dd,ii=cKDTree(Vg).query(V2); print('simp: verts',len(V2),'faces',len(F2),'max match dist',dd.max())
    assert dd.max()<1e-4
    Vg=Vg[ii];Cg=Cg[ii];part=part[ii];Wf=Wf[ii];Fg=F2
mesh=bpy.data.meshes.new('Her')
Vbl=np.c_[Vg[:,0],-Vg[:,2],Vg[:,1]]
mesh.from_pydata(Vbl.tolist(),[],Fg.tolist()); mesh.update()
body=bpy.data.objects.new('Her',mesh); bpy.context.scene.collection.objects.link(body)
ca=mesh.color_attributes.new('Col','FLOAT_COLOR','POINT')
cc=np.c_[np.clip(Cg,0,255)/255.0,np.ones(len(Cg))]
ca.data.foreach_set('color',cc.ravel())
for p in mesh.polygons: p.use_smooth=True
# material using vertex colors
mat=bpy.data.materials.new('M'); mat.use_nodes=True
nt=mat.node_tree; bsdf=nt.nodes['Principled BSDF']; vc=nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name='Col'
nt.links.new(vc.outputs['Color'],bsdf.inputs['Base Color']); bsdf.inputs['Roughness'].default_value=0.6
mesh.materials.append(mat)
body.parent=arm
m_=body.modifiers.new('Armature','ARMATURE'); m_.object=arm
for j,nm in enumerate(names):
    vg=body.vertex_groups.new(name=nm)
    ids=np.where(Wf[:,j]>1e-4)[0]
    for i in ids: vg.add([int(i)],float(Wf[i,j]),'REPLACE')
Vb=Vg   # gltf coords, lifted
# ---------------- gait helpers ----------------
def sag(v): return math.atan2(-v[1],-v[2])
_u=arm.data.bones['UpperArm.L']; _l=arm.data.bones['LowerArm.L']
REST_ELB=sag(_l.tail_local-_l.head_local)-sag(_u.tail_local-_u.head_local)     # her base pose has the forearm already bent forward
print('rest elbow flexion deg',math.degrees(REST_ELB))
FPS=30
bpy.context.scene.render.fps=FPS
bpy.context.view_layer.objects.active=arm; bpy.ops.object.mode_set(mode='POSE')
pb=arm.pose.bones
for b in pb: b.rotation_mode='QUATERNION'
Rb={b.name:b.bone.matrix_local.to_3x3() for b in pb}
def rotX(a): return Matrix.Rotation(a,3,'X')
def rotY(a): return Matrix.Rotation(a,3,'Y')
def rotZ(a): return Matrix.Rotation(a,3,'Z')
def lean_rot(s): return rotX(-s)
def setrot(name,c): pb[name].rotation_quaternion=(Rb[name].inverted()@c@Rb[name]).to_quaternion()
def sm(t): return t*t*(3-2*t)
AY=JJ['l-ankle'][1]; AZ=JJ['l-ankle'][2]
def leg_geom(sd):
    s=1 if sd=='L' else -1; m=np.array([s,1,1.])
    return JJ['l-upper-leg']*m,JJ['l-knee']*m,JJ['l-ankle']*m,np.array([0.18*s,0.0475,0.205])
def ang(dy,dz): return math.atan2(dz,-dy)
def vec(angle,L): return np.array([-L*math.cos(angle),L*math.sin(angle)])
rest_t1={}
for sd in 'LR':
    h,k,a,t=leg_geom(sd); rest_t1[sd]=(ang(*(k-h)[1:]),ang(*(a-k)[1:]),ang(*(t-a)[1:]))
LEGLEN=HB-AY
SCALE=LEGLEN/0.78            # relative to the reference male rig that the gait constants were tuned on
# rest leg tilt (hip->ankle lateral) in degrees
REST_TILT=math.degrees(math.atan2(JJ['l-ankle'][0]-JJ['l-upper-leg'][0],LEGLEN))
print('leg length',LEGLEN,'scale',SCALE,'rest tilt',REST_TILT)
PHI_W=math.radians(REST_TILT+FEM['walk_ankle_in']); PHI_R=math.radians(REST_TILT+FEM['run_ankle_in'])
# ---------------- WALK ----------------
CYC=FEM['walk_cycle_frames']; NCYC=2; NF=CYC*NCYC
SL=0.20*SCALE*FEM['stride_factor']
def leg_pose(sd,ph,hipy):
    h,k,a,t=leg_geom(sd); L1=np.linalg.norm((k-h)[1:]); L2=np.linalg.norm((a-k)[1:])
    ST=0.6
    if ph<ST:
        u=ph/ST; z=SL-2*SL*u; pitch=math.radians(12*(1-u)**1.5-14*u**3); y=0.045*max(0,(u-0.7)/0.3)**1.5
    else:
        u=(ph-ST)/(1-ST); e=sm(u); z=-SL+2*SL*e; y=0.045*(1-sm(min(1,u*3)))+0.07*math.sin(math.pi*u)**1.2
        pitch=math.radians(-14*(1-sm(min(1,u*1.6)))+12*sm(max(0,(u-0.5)/0.5)))
    tgt_y=AY+y; tgt_z=AZ+z
    dy=tgt_y-hipy; dz=tgt_z-HZ0
    d=min(math.hypot(dy,dz),(L1+L2)*0.999)
    A_=math.acos(max(-1,min(1,(L1*L1+d*d-L2*L2)/(2*L1*d)))); line=ang(dy,dz)
    t1=line+A_
    kn=np.array(vec(t1,L1)); ank=np.array([dy,dz]); sh=ank-kn; t2=ang(sh[0],sh[1])
    return t1,t2,pitch
W_=FEM['walk']
def pose_walk(f):
    ph=(f%CYC)/CYC
    hipy=HB-0.030*SCALE-0.012*math.cos(4*math.pi*ph)
    swayx=0.016*math.sin(2*math.pi*(ph-0.05))*FEM['walk_sway']
    pb['Hips'].location=Rb['Hips'].inverted()@Vector((swayx,0,hipy-HB))
    for sd in 'LR':
        s=1 if sd=='L' else -1
        pp=(ph if sd=='L' else (ph+0.5)%1.0)
        t1,t2,pitch=leg_pose(sd,pp,hipy)
        r1,r2,r3=rest_t1[sd]
        tw=t1-r1; sw_=t2-r2
        setrot('UpperLeg.'+sd,lean_rot(tw)@rotY(PHI_W*s))
        setrot('LowerLeg.'+sd,lean_rot(sw_-tw))
        setrot('Foot.'+sd,lean_rot(pitch-sw_)@rotY(-PHI_W*s))
    yaw=math.radians(W_['yaw'])*math.sin(2*math.pi*ph)
    roll=math.radians(W_['roll'])*math.sin(2*math.pi*(ph-0.05))
    setrot('Hips',rotZ(-yaw)@rotY(roll))
    setrot('Spine',rotZ(+yaw*W_['spine_yaw'])@rotY(-roll*W_['spine_roll']))
    setrot('Chest',rotZ(+yaw*W_['chest_yaw'])@rotY(-roll*W_['chest_roll']))
    setrot('Neck',rotZ(-yaw*0.35)); setrot('Head',rotZ(-yaw*0.35))
    for sd in 'LR':
        s=1 if sd=='L' else -1
        pp=(ph if sd=='L' else (ph+0.5)%1.0)
        swing=-math.radians(W_['arm_swing'])*math.cos(2*math.pi*pp)
        add=Matrix.Rotation(math.radians(W_['arm_add'])*s,3,'Y')
        setrot('UpperArm.'+sd,lean_rot(swing)@add)
        elb=math.radians(14)+math.radians(14)*(0.5-0.5*math.cos(2*math.pi*pp))
        setrot('LowerArm.'+sd,lean_rot(elb-REST_ELB)@Matrix.Rotation(math.radians(W_['elbow_add'])*s,3,'Y'))
        setrot('Hand.'+sd,Matrix.Identity(3))
arm.animation_data_create()
WALK=bpy.data.actions.new('Walk'); arm.animation_data.action=WALK
for f in range(NF+1):
    pose_walk(f)
    for b in pb:
        b.keyframe_insert('rotation_quaternion',frame=f+1)
        if b.name=='Hips': b.keyframe_insert('location',frame=f+1)
tr_=arm.animation_data.nla_tracks.new(); tr_.name='Walk'; tr_.strips.new('Walk',1,WALK); arm.animation_data.action=None
# ---------------- RUN ----------------
for b in pb: b.rotation_quaternion=(1,0,0,0); b.location=(0,0,0)
RUN_ACT=bpy.data.actions.new('Run'); arm.animation_data.action=RUN_ACT
RCYC=FEM['run_cycle_frames']; RNF=RCYC
ST=0.36
ZF=SCALE*FEM['stride_factor']
sw=lambda u:ST+(1-ST)*u
K=[(0.00,0.12,0.112,8),(0.09,0.035,0.100,2),(0.18,-0.05,0.108,-2),(0.27,-0.135,0.130,-8),(ST,-0.22,0.176,-22),
   (sw(0.15),-0.34,0.34,-36),(sw(0.35),0.00,0.55,-30),(sw(0.55),0.30,0.46,-20),(sw(0.75),0.40,0.30,-10),(sw(0.90),0.26,0.16,3),(1.0,0.12,0.112,8)]
kp=np.array([k[0] for k in K]); fz=PchipInterpolator(kp,[k[1]*ZF for k in K]); fy=PchipInterpolator(kp,[k[2] for k in K]); fp=PchipInterpolator(kp,[math.radians(k[3]) for k in K])
HIP_MIN=HB-0.07*SCALE; HIP_A,HIP_PH=0.13,0.12; HIPZ=-0.05
def run_ik(sd,ph,hipy,dy_corr=0.0):
    h,k,a,t=leg_geom(sd); L1=np.linalg.norm((k-h)[1:]); L2=np.linalg.norm((a-k)[1:])
    tgt_y=float(fy(ph))+dy_corr; tgt_z=AZ+float(fz(ph)); pitch=float(fp(ph))
    dy=tgt_y-hipy; dz=tgt_z-(HZ0+HIPZ)
    d=min(math.hypot(dy,dz),(L1+L2)*0.999)
    A_=math.acos(max(-1,min(1,(L1*L1+d*d-L2*L2)/(2*L1*d)))); line=ang(dy,dz)
    t1=line+A_; kn=np.array(vec(t1,L1)); sh=np.array([dy,dz])-kn; t2=ang(sh[0],sh[1])
    return t1,t2,pitch
LEAN_H,LEAN_S,LEAN_C=math.radians(4.0),math.radians(3.5),math.radians(3.0); CUM=LEAN_H+LEAN_S+LEAN_C
R_=FEM['run']
corr={}
def pose_run(f):
    ph=(f%RCYC)/RCYC
    hipy=HIP_MIN+HIP_A*(0.5-0.5*math.cos(4*math.pi*(ph-HIP_PH)))
    swayx=0.010*math.sin(2*math.pi*(ph-0.10))*FEM['run_sway']
    pb['Hips'].location=Rb['Hips'].inverted()@Vector((swayx,-HIPZ,hipy-HB))
    for sd in 'LR':
        s=1 if sd=='L' else -1
        pp=(ph if sd=='L' else (ph+0.5)%1.0)
        t1,t2,pitch=run_ik(sd,pp,hipy,corr.get((f%RCYC,sd),0.0))
        r1,r2,r3=rest_t1[sd]; tw=t1-r1; sw_=t2-r2
        setrot('UpperLeg.'+sd,lean_rot(tw-LEAN_H)@rotY(PHI_R*s))
        setrot('LowerLeg.'+sd,lean_rot(sw_-tw))
        setrot('Foot.'+sd,lean_rot(pitch-sw_)@rotY(-PHI_R*s))
    yaw=math.radians(R_['yaw'])*math.sin(2*math.pi*(ph-0.35))
    bob=math.radians(1.5)*math.cos(4*math.pi*(ph-HIP_PH))
    roll=math.radians(R_['roll'])*math.sin(2*math.pi*(ph-0.10))
    setrot('Hips',rotZ(-yaw)@rotY(roll)@lean_rot(-LEAN_H))
    setrot('Spine',rotZ(+yaw*R_['spine_yaw'])@rotY(-roll*R_['spine_roll'])@lean_rot(-(LEAN_S+bob)))
    setrot('Chest',rotZ(+yaw*R_['chest_yaw'])@rotY(-roll*R_['chest_roll'])@lean_rot(-LEAN_C))
    setrot('Neck',rotZ(-yaw*0.3)@lean_rot(0.4*(CUM+bob)))
    setrot('Head',rotZ(-yaw*0.5)@lean_rot(0.6*(CUM+bob)))
    for sd in 'LR':
        s=1 if sd=='L' else -1
        pp=(ph if sd=='L' else (ph+0.5)%1.0)
        c=math.cos(2*math.pi*(pp-0.40))
        swing=math.radians(6)+math.radians(R_['arm_swing'])*c
        add=Matrix.Rotation(math.radians(R_['arm_add'])*s,3,'Y')
        setrot('UpperArm.'+sd,lean_rot(swing-CUM)@add)
        elb=math.radians(88)+math.radians(10)*c
        setrot('LowerArm.'+sd,lean_rot(elb-REST_ELB)@Matrix.Rotation(math.radians(R_['elbow_add'])*s,3,'Y'))
        setrot('Hand.'+sd,lean_rot(0))
footsel={sd:np.where((Vb[:,1]<0.05)&((Vb[:,0]>0.04) if sd=='L' else (Vb[:,0]<-0.04)))[0] for sd in 'LR'}
def foot_min(sd):
    bpy.context.view_layer.update()
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh()
    co=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',co); co=co.reshape(-1,3); e.to_mesh_clear()
    return co[footsel[sd],2].min()
GROUND=-0.005 if False else 0.0
for it in range(4):
    worst=0
    for f in range(RCYC):
        pose_run(f)
        for sd in 'LR':
            pp=((f/RCYC) if sd=='L' else ((f/RCYC)+0.5)%1.0)
            m=foot_min(sd); err=m-GROUND
            if pp<0.28: corr[(f,sd)]=corr.get((f,sd),0.0)-0.9*err; worst=max(worst,abs(err))
            elif err<0: corr[(f,sd)]=corr.get((f,sd),0.0)-0.9*err
    print('ground-plant iteration',it,'worst stance error mm',round(1000*worst,1))
for f in range(RNF+1):
    pose_run(f)
    for b in pb:
        b.keyframe_insert('rotation_quaternion',frame=f)
        if b.name=='Hips': b.keyframe_insert('location',frame=f)
tr2=arm.animation_data.nla_tracks.new(); tr2.name='Run'; tr2.strips.new('Run',0,RUN_ACT); arm.animation_data.action=None
bpy.context.scene.frame_start=1; bpy.context.scene.frame_end=NF+1
# verify walk ground contact too
for b in pb: b.rotation_quaternion=(1,0,0,0); b.location=(0,0,0)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('SAVED',OUT)
