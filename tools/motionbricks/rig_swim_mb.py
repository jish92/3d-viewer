import bpy,sys,numpy as np,math
from mathutils import Matrix,Quaternion
from scipy.spatial.transform import Rotation as Rot
OUT=sys.argv[sys.argv.index('--')+1]; CLIP=sys.argv[sys.argv.index('--')+2]
bpy.ops.wm.open_mainfile(filepath='/workspace/me3d/swim/brief_run_rigged.blend')
d=np.load(CLIP); Qm=d['Qm']; H=d['hips']; names=list(d['bones']); n=len(Qm)
q=Rot.from_matrix(Qm.reshape(-1,3,3)).as_quat().reshape(n,len(names),4)[...,[3,0,1,2]]   # wxyz
for i in range(1,n):
    s=np.sign((q[i]*q[i-1]).sum(-1,keepdims=True)); q[i]*=np.where(s==0,1,s)
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
ms=[o for o in bpy.data.objects if o.type=='MESH']; body=max(ms,key=lambda o:len(o.data.vertices))
bpy.context.view_layer.objects.active=arm; bpy.ops.object.mode_set(mode='POSE')
pb=arm.pose.bones
for t in arm.animation_data.nla_tracks: t.mute=True
# vertex set for the feet (rest pose)
arm.data.pose_position='REST'; bpy.context.view_layer.update()
def verts():
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh(); co=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',co); co=co.reshape(-1,3); e.to_mesh_clear(); return co
V0=verts(); arm.data.pose_position='POSE'
fs={s:np.where((V0[:,2]<0.22)&(V0[:,0]*s>0.04))[0] for s in (1,-1)}
ACT=bpy.data.actions.new('MotionBricks'); arm.animation_data.action=ACT
def key(dz):
    for f in range(n):
        for i,nm in enumerate(names):
            b=pb[nm]; b.rotation_quaternion=Quaternion(q[f,i]); b.keyframe_insert('rotation_quaternion',frame=f)
        h=pb['Hips']; rest=arm.data.bones['Hips']; Rr=np.array(rest.matrix_local)[:3,:3]
        loc=Rr.T@(H[f]+np.array([0,0,dz])-np.array(rest.head_local)); h.location=loc; h.keyframe_insert('location',frame=f)
def lows():
    out=[]
    for f in range(n):
        bpy.context.scene.frame_set(f); V=verts(); out.append((V[fs[1],2].min(),V[fs[-1],2].min()))
    return np.array(out)
key(0.0); L=lows(); lo=L.min(1); np.save('/tmp/lows0.npy',L)
print('sole min per frame: min %.3f p5 %.3f median %.3f max %.3f'%(lo.min(),np.percentile(lo,5),np.median(lo),lo.max()))
dz=-0.005-np.percentile(lo,3); print('ground dz',dz); key(dz); L=lows(); lo=L.min(1)
print('after: min %.3f p5 %.3f median %.3f max %.3f; both-feet-off frames (>2cm):'%(lo.min(),np.percentile(lo,5),np.median(lo),lo.max()),int((lo>0.02).sum()),'of',n)
for fc in []:
    for kp in fc.keyframe_points: kp.interpolation='LINEAR'
tr=arm.animation_data.nla_tracks.new(); tr.name='MotionBricks'; tr.strips.new('MotionBricks',0,ACT); arm.animation_data.action=None
for t in arm.animation_data.nla_tracks: t.mute=False
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=OUT.replace('.glb','.blend'))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',use_selection=True,export_animations=True,export_skins=True,
    export_force_sampling=True,export_frame_range=False,export_animation_mode='ACTIONS',export_yup=True,export_vertex_color='ACTIVE',
    export_apply=False,export_image_format='AUTO',export_optimize_animation_size=False)
# head rigidity report
hd=pb['Head']; 
import mathutils
print('EXPORTED',OUT)
