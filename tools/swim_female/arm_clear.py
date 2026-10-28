"""Arm clearance pass: for every frame of every action add the smallest extra upper-arm abduction (rotation about the chest's forward axis, plus part of it at the elbow)
so that no forearm/hand/upper-arm vertex is inside the torso/hips/thighs (and the forearm/hand keep >= MINGAP from them). Temporally smoothed (periodic), then re-verified.
usage: blender --python arm_clear.py -- in.blend out.blend"""
import bpy,sys,numpy as np,math,os
from mathutils import Matrix,Quaternion
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d,maximum_filter1d
IN,OUT=sys.argv[sys.argv.index('--')+1:sys.argv.index('--')+3]
MARGIN=float(os.environ.get('MARGIN','0.004')); PENTOL=float(os.environ.get('PENTOL','0.002')); EMAX=math.radians(float(os.environ.get('EMAX','40'))); STEP=math.radians(1.0); ELB=float(os.environ.get('ELB_FRAC','0.35'))
bpy.ops.wm.open_mainfile(filepath=IN)
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
ms=[o for o in bpy.data.objects if o.type=='MESH']; body=max(ms,key=lambda o:len(o.data.vertices))
nv=len(body.data.vertices); gname={g.index:g.name for g in body.vertex_groups}
dom=np.full(nv,-1); wmax=np.zeros(nv)
for v in body.data.vertices:
    for g in v.groups:
        if g.weight>wmax[v.index]: wmax[v.index]=g.weight; dom[v.index]=g.group
dn=np.array([gname.get(d,'') for d in dom])
isarm=np.array([n.startswith(('LowerArm','Hand','UpperArm')) for n in dn]); isfore=np.array([n.startswith(('LowerArm','Hand')) for n in dn])
rest=np.empty(nv*3); body.data.vertices.foreach_get('co',rest); rest=rest.reshape(-1,3)
shL=np.array(arm.data.bones['UpperArm.L'].head_local); shR=np.array(arm.data.bones['UpperArm.R'].head_local)
dsh=np.minimum(np.linalg.norm(rest-shL,axis=1),np.linalg.norm(rest-shR,axis=1))
test=isfore|(isarm&(dsh>0.10)); tgt=np.where(~isarm&(dn!='Head')&(dn!='Neck'))[0]
side={s:np.where(test&np.array([n.endswith('.'+s) for n in dn]))[0] for s in 'LR'}
ARMB=['UpperArm.L','UpperArm.R']
Rb={b.name:b.bone.matrix_local.to_3x3() for b in arm.pose.bones}
pb=arm.pose.bones
for b in pb: b.rotation_mode='QUATERNION'
def evalmesh():
    bpy.context.view_layer.update()
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh()
    co=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',co); co=co.reshape(-1,3)
    nn=np.empty(len(m.vertices)*3); m.vertices.foreach_get('normal',nn); nn=nn.reshape(-1,3); e.to_mesh_clear(); return co,nn
def abd(name,e): s=1 if name.endswith('.L') else -1; return Rb[name].inverted()@Matrix.Rotation(-e*s,3,'Y')@Rb[name]   # parent-space abduction (+e = away from the body)
def apply(q0,eL,eR):
    for sd,e in (('L',eL),('R',eR)):
        un='UpperArm.'+sd; ln='LowerArm.'+sd
        pb[un].rotation_quaternion=(abd(un,e)@Quaternion(q0[un]).to_matrix()).to_quaternion()
        pb[ln].rotation_quaternion=(abd(ln,ELB*e)@Quaternion(q0[ln]).to_matrix()).to_quaternion()
def viol():
    co,nn=evalmesh(); kd=cKDTree(co[tgt]); out={}
    for sd in 'LR':
        ix=side[sd]; P=co[ix]; d,i=kd.query(P); q=tgt[i]; sdist=((P-co[q])*nn[q]).sum(1)
        near=d<0.06; f_=isfore[ix]
        pen_up=np.where(near&~f_&(sdist<0),-sdist,0).max()                       # upper arm: no penetration
        low_fore=np.where(near&f_,sdist,9).min()                                  # forearm/hand: keep MARGIN outside the surface
        out[sd]=(pen_up>PENTOL) or (low_fore<MARGIN)
    return out
# clear existing animation drive
for t in arm.animation_data.nla_tracks: t.mute=True
arm.animation_data.action=None
res={}
for tr in list(arm.animation_data.nla_tracks):
    if os.environ.get('ONLY') and tr.name!=os.environ['ONLY']: continue
    act=tr.strips[0].action; FC=[fc for l in act.layers for st in l.strips for cb in st.channelbags for fc in cb.fcurves]; fcs={(fc.data_path,fc.array_index):fc for fc in FC}
    frames=sorted({int(round(k.co[0])) for fc in FC for k in fc.keyframe_points}); n=len(frames)
    def q_at(f): return {b.name:([fcs[('pose.bones["%s"].rotation_quaternion'%b.name,i)].evaluate(f) for i in range(4)] if ('pose.bones["%s"].rotation_quaternion'%b.name,0) in fcs else [1,0,0,0]) for b in pb}
    def loc_at(f): return [fcs[('pose.bones["Hips"].location',i)].evaluate(f) for i in range(3)]
    E=np.zeros((n,2)); Q0=[]
    for k,f in enumerate(frames):
        q0=q_at(f); Q0.append(q0)
        for b in pb: b.rotation_quaternion=q0[b.name]
        pb['Hips'].location=loc_at(f)
        e=[0.0,0.0]
        for it in range(int(EMAX/STEP)+1):
            apply(q0,e[0],e[1]); v=viol()
            if not (v['L'] or v['R']): break
            for j,sd in enumerate('LR'):
                if v[sd]: e[j]+=STEP
        E[k]=e
    raw=E.copy()
    # periodic smoothing that never goes below the required abduction
    Es=np.zeros_like(E)
    for j in range(2):
        m=maximum_filter1d(E[:,j],5,mode='wrap'); g=gaussian_filter1d(m,1.6,mode='wrap'); Es[:,j]=np.maximum(E[:,j],g)
    # verify and bump where the smoothed value is not enough (should not happen)
    for rnd in range(3):
        bad=0
        for k,f in enumerate(frames):
            for b in pb: b.rotation_quaternion=Q0[k][b.name]
            pb['Hips'].location=loc_at(f); apply(Q0[k],Es[k,0],Es[k,1]); v=viol()
            for j,sd in enumerate('LR'):
                if v[sd]: Es[k,j]+=STEP; bad+=1
        if not bad: break
    res[tr.name]=(frames,Es)
    print('%s: frames %d  extra abduction deg  L mean %.1f max %.1f | R mean %.1f max %.1f   (raw max %.1f %.1f)'%(tr.name,n,*np.degrees([Es[:,0].mean(),Es[:,0].max(),Es[:,1].mean(),Es[:,1].max(),raw[:,0].max(),raw[:,1].max()])))
    # write the new quaternions into the action
    prev={}
    for k,f in enumerate(frames):
        for sd,e in (('L',Es[k,0]),('R',Es[k,1])):
            for nm,fr in (('UpperArm.'+sd,1.0),('LowerArm.'+sd,ELB)):
                q=(abd(nm,fr*e)@Quaternion(Q0[k][nm]).to_matrix()).to_quaternion()
                if nm in prev and (np.array(q)@np.array(prev[nm]))<0: q=-q
                prev[nm]=q
                for i in range(4):
                    fc=fcs[('pose.bones["%s"].rotation_quaternion'%nm,i)]
                    for kp in fc.keyframe_points:
                        if int(round(kp.co[0]))==f: kp.co[1]=q[i]; kp.handle_left[1]=q[i]; kp.handle_right[1]=q[i]
for t in arm.animation_data.nla_tracks: t.mute=False
for b in pb: b.rotation_quaternion=(1,0,0,0); b.location=(0,0,0)
bpy.ops.wm.save_as_mainfile(filepath=OUT); print('SAVED',OUT)
