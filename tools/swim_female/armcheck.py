# usage: blender-python armcheck.py -- file.blend [tag]   -> per-frame arm-vs-body penetration for every NLA action
import bpy,sys,numpy as np,json
from scipy.spatial import cKDTree
BL=sys.argv[sys.argv.index('--')+1]; TAG=sys.argv[sys.argv.index('--')+2] if len(sys.argv)>sys.argv.index('--')+2 else ''
bpy.ops.wm.open_mainfile(filepath=BL)
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
ms=[o for o in bpy.data.objects if o.type=='MESH']; body=max(ms,key=lambda o:len(o.data.vertices))
nv=len(body.data.vertices); vg=body.vertex_groups
dom=np.full(nv,-1); wmax=np.zeros(nv)
for v in body.data.vertices:
    for g in v.groups:
        if g.weight>wmax[v.index]: wmax[v.index]=g.weight; dom[v.index]=g.group
gname={g.index:g.name for g in vg}
isarm=np.array([gname.get(d,'').startswith(('LowerArm','Hand','UpperArm')) for d in dom])
isfore=np.array([gname.get(d,'').startswith(('LowerArm','Hand')) for d in dom])
# shoulder region excluded for upper-arm verts
rest=np.empty(nv*3); body.data.vertices.foreach_get('co',rest); rest=rest.reshape(-1,3)
shL=np.array(arm.data.bones['UpperArm.L'].head_local); shR=np.array(arm.data.bones['UpperArm.R'].head_local)
dsh=np.minimum(np.linalg.norm(rest-shL,axis=1),np.linalg.norm(rest-shR,axis=1))
test=isfore|(isarm&(dsh>0.10))
tgt=np.where(~isarm & (np.array([gname.get(d,'') for d in dom])!='Head') & (np.array([gname.get(d,'') for d in dom])!='Neck'))[0]
print('arm test verts',test.sum(),'body verts',len(tgt))
res={}
def evalmesh():
    bpy.context.view_layer.update()
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh()
    co=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',co); co=co.reshape(-1,3)
    nn=np.empty(len(m.vertices)*3); m.vertices.foreach_get('normal',nn); nn=nn.reshape(-1,3); e.to_mesh_clear(); return co,nn
tracks=list(arm.animation_data.nla_tracks)
ARM=arm.animation_data
for tr in tracks:
    for t in tracks: t.mute=(t.name!=tr.name)
    st=tr.strips[0]; f0=int(st.frame_start); f1=int(st.frame_end)
    out=[]
    for f in range(f0,f1+1):
        bpy.context.scene.frame_set(f)
        co,nn=evalmesh()
        kd=cKDTree(co[tgt]); P=co[test]
        d,i=kd.query(P)
        q=tgt[i]; sd=((P-co[q])*nn[q]).sum(1)
        ins=(sd<0)&(d<0.06)
        pen=np.where(ins,-sd,0); k=np.argmax(pen)
        # also the arm-to-body clearance (min distance) for forearm/hand
        out.append((f,float(pen.max()),int((pen>0.002).sum()),float(d.min()),int(np.where(test)[0][k]),gname.get(dom[np.where(test)[0][k]],'')))
    arr=np.array([o[1] for o in out]); cnt=np.array([o[2] for o in out]); dmin=np.array([o[3] for o in out])
    print('%s %s frames %d  max pen mm %.1f  frames with >2mm pen: %d  verts>2mm max %d  min dist mm %.1f'%(TAG,tr.name,len(out),1000*arr.max(),int((arr>0.002).sum()),cnt.max(),1000*dmin.min()))
    bad=[o for o in out if o[1]>0.002]
    for o in bad[:12]: print('   frame',o[0],'pen mm %.1f verts %d bone %s'%(1000*o[1],o[2],o[5]))
    res[tr.name]=out
json.dump(res,open('/tmp/armcheck_%s.json'%TAG,'w'))
