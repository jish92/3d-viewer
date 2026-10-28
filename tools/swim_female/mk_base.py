import sys,os,json
sys.path.insert(0,'/workspace/me3d/swim_female/ext')
import bpy,numpy as np
UD='/workspace/me3d/swim_female/mpfb_user'
bpy.utils.extension_path_user=lambda pkg,path='',create=False: os.path.join(UD,path) if path else UD
import mpfb; mpfb.register()
from mpfb.services.humanservice import HumanService
from mpfb.services.targetservice import TargetService
bpy.ops.wm.read_factory_settings(use_empty=True)
P=json.load(open('/workspace/me3d/swim_female/params.json'))
d=TargetService.get_default_macro_info_dict()
d.update(P['macro']); d['race']=P['race']
h=HumanService.create_human(mask_helpers=False,detailed_helpers=True,extra_vertex_groups=True,feet_on_ground=False,scale=0.1,macro_detail_dict=d)
bpy.context.view_layer.objects.active=h
for name,w in P['details'].items():
    for side in ([''] if not name.startswith('L/') else ['l-','r-']):
        n=name[2:] if name.startswith('L/') else name
        n=n.replace('SIDE',side[:-1] if side else '') if 'SIDE' in n else (side+n)
        fp=TargetService.target_full_path(n)
        if not fp: print('MISSING',n); continue
        TargetService.load_target(h,fp,weight=w)
bpy.context.view_layer.update()
h.shape_key_add(name='mix',from_mix=True)
for k in list(h.data.shape_keys.key_blocks)[:-1]: h.shape_key_remove(k)
h.shape_key_remove(h.data.shape_keys.key_blocks[0])
A='/workspace/me3d/swim_female/assets/'
extra={}
for key,mh in [('eyes',A+'eyes/low-poly/low-poly.mhclo'),('brow',A+'eyebrows/eyebrow001/eyebrow001.mhclo'),('lash',A+'eyelashes/eyelashes01/eyelashes01.mhclo')]:
    o=HumanService.add_mhclo_asset(mh,h,asset_type='Eyes' if key=='eyes' else ('Eyebrows' if key=='brow' else 'Eyelashes'),subdiv_levels=0,set_up_rigging=False,import_weights=False)
    me=o.data
    uv=me.uv_layers.active.data if me.uv_layers.active else None
    uvv=np.zeros((len(me.vertices),2))
    if uv:
        for l in me.loops: uvv[l.vertex_index]=uv[l.index].uv
    sh=o.matrix_world
    extra[key+'_V']=np.array([(sh@v.co)[:] for v in me.vertices]); extra[key+'_P']=np.array([list(p.vertices) for p in me.polygons],dtype=object); extra[key+'_UV']=uvv
    print(key,len(me.vertices),o.name)
np.savez('/workspace/me3d/swim_female/extra_raw.npz',allow_pickle=True,**extra)
V=np.array([v.co[:] for v in h.data.vertices])
polys=[list(p.vertices) for p in h.data.polygons]
groups={}
for g in h.vertex_groups:
    groups[g.name]=[]
for v in h.data.vertices:
    for ge in v.groups:
        groups[h.vertex_groups[ge.group].name].append((v.index,ge.weight))
bi=set(i for i,w in groups['body'] if w>.5)
np.savez('/workspace/me3d/swim_female/full_raw.npz',V=V,polys=np.array(polys,dtype=object),allow_pickle=True)
json.dump({k:v for k,v in groups.items()},open('/workspace/me3d/swim_female/groups.json','w'))
print(V.shape,len(polys),len(bi))
