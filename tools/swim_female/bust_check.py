import bpy,numpy as np,sys,json
from mathutils import Vector
bpy.ops.wm.open_mainfile(filepath='/workspace/me3d/swim_female/out/her_final.blend')
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]; body=[o for o in bpy.data.objects if o.type=='MESH'][0]
pb=arm.pose.bones
tracks={t.name:t for t in arm.animation_data.nla_tracks}
vg={g.name:g.index for g in body.vertex_groups}
nV=len(body.data.vertices)
Wb={sd:np.zeros(nV) for sd in 'LR'}
for v in body.data.vertices:
    for g in v.groups:
        for sd in 'LR':
            if g.group==vg['Bust.'+sd]: Wb[sd][v.index]=g.weight
print('bust-weighted verts',{k:int((w>1e-3).sum()) for k,w in Wb.items()},'max w',{k:round(w.max(),2) for k,w in Wb.items()})
def verts():
    bpy.context.view_layer.update()
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh(); co=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',co); co=co.reshape(-1,3)
    F=np.array([p.vertices[:] for p in m.polygons]); e.to_mesh_clear(); return co,F
def vw(sd):
    base=pb['Chest'].matrix@(pb['Chest'].bone.matrix_local.inverted()@pb['Bust.'+sd].bone.matrix_local)
    return np.array(pb['Bust.'+sd].matrix.translation-base.translation)
out={}
for name in ('Run','MotionBricks','Walk'):
    for n,t in tracks.items(): t.mute=(n!=name)
    st=tracks[name].strips[0]; best=(0,None)
    for f in range(int(st.frame_start),int(st.frame_end)+1):
        bpy.context.scene.frame_set(f); m=max(np.linalg.norm(vw(s)) for s in 'LR')
        if m>best[0]: best=(m,f)
    f=best[1]; bpy.context.scene.frame_set(f); A,F=verts()
    B=A-sum(Wb[s][:,None]*vw(s)[None] for s in 'LR')
    E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); E.sort(1); E=np.unique(E,axis=0)
    la=np.linalg.norm(A[E[:,0]]-A[E[:,1]],axis=1); lb=np.linalg.norm(B[E[:,0]]-B[E[:,1]],axis=1)
    ok=lb>1e-4; strain=np.abs(la[ok]/lb[ok]-1); disp=np.linalg.norm(A-B,axis=1)
    if name=='Run':
        ii=np.argsort(-np.abs(la-lb))[:6]
        for i in ii: print('EDGE',E[i],'len mm',round(lb[i]*1000,1),'pos',B[E[i,0]].round(3),'wb',round(Wb['L'][E[i,0]]+Wb['R'][E[i,0]],2),round(Wb['L'][E[i,1]]+Wb['R'][E[i,1]],2))
    out[name]=dict(frame=int(f),bone_offset_mm=round(float(best[0])*1000,2),max_vertex_disp_mm=round(float(disp.max())*1000,2),max_edge_change_mm=round(float(np.abs(la-lb).max())*1000,3),mean_edge_len_in_bust_mm=round(float(lb[(np.abs(la-lb)>1e-6)].mean())*1000,2),max_edge_strain_pct=round(float(strain.max())*100,2),p99_edge_strain_pct=round(float(np.percentile(strain,99))*100,3),moved_verts=int((disp>1e-5).sum()))
    print(name,out[name])
json.dump(out,open('/workspace/me3d/swim_female/out/bust_check.json','w'))
