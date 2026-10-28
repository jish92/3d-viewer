import sys,os
sys.path.insert(0,'/workspace/me3d/swim_female/ext')
import bpy,bmesh,numpy as np
bpy.ops.wm.open_mainfile(filepath='/workspace/me3d/swim_female/base.blend')
h=bpy.data.objects['Human']
# apply shape keys
bpy.context.view_layer.objects.active=h
h.shape_key_add(name='mix',from_mix=True)
for k in list(h.data.shape_keys.key_blocks)[:-1]: h.shape_key_remove(k)
h.shape_key_remove(h.data.shape_keys.key_blocks[0])
gi=h.vertex_groups['body'].index
keep=np.array([any(g.group==gi and g.weight>0.5 for g in v.groups) for v in h.data.vertices])
V=np.array([v.co[:] for v in h.data.vertices]); 
# faces
F=[tuple(p.vertices) for p in h.data.polygons if all(keep[i] for i in p.vertices)]
idx=-np.ones(len(V),int); idx[keep]=np.arange(keep.sum())
F=[tuple(idx[i] for i in f) for f in F]
V=V[keep]
# vertex groups kept for later?
np.savez('/workspace/me3d/swim_female/body_raw.npz',V=V,F=np.array([f for f in F if len(f)==4]),T=np.array([f for f in F if len(f)==3]) if any(len(f)==3 for f in F) else np.zeros((0,3),int))
print(V.shape,len(F),V.min(0),V.max(0))
# also uv
