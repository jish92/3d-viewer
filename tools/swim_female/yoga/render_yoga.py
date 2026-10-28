# usage: python render_yoga.py -- file.blend TrackName az el dist tz out_prefix [x_target]
import bpy,sys,math,os
sys.path.insert(0,'/workspace/me3d/swim_female/yoga')
exec(open('/workspace/me3d/swim_female/yoga/ybl.py').read())
a=sys.argv[sys.argv.index('--')+1:]; BL,TR,AZ,EL,DIST,TZ,OUT=a[:7]; TX=float(a[7]) if len(a)>7 else 0.0; TY=float(a[8]) if len(a)>8 else 0.0
bpy.ops.wm.open_mainfile(filepath=BL)
arm=get_arm(); body=get_body()
for t in arm.animation_data.nla_tracks: t.mute=(t.name!=TR)
st=[t for t in arm.animation_data.nla_tracks if t.name==TR][0].strips[0]
setup_scene(floor=True)
if os.environ.get('SMILE'):
    kb,inf=add_smile(body,amt=float(os.environ.get('SMILEAMT','1.0'))); kb.value=1.0; print('SMILE',inf)
sc=bpy.context.scene; sc.render.fps=30; sc.frame_start=int(st.frame_start); sc.frame_end=int(st.frame_end)-0
w=sc.world; w.color=(0.80,0.80,0.82)
fl=bpy.data.objects['Floor']; fl.data.color_attributes.new('Col','FLOAT_COLOR','POINT')
import numpy as np,os
ca=fl.data.color_attributes['Col']; ca.data.foreach_set('color',np.tile([float(os.environ.get('FC','2.0'))]*3+[1.0],len(fl.data.vertices)))
fl.data.color_attributes.active_color=ca; fl.data.color_attributes.render_color_index=list(fl.data.color_attributes).index(ca)
sh=sc.display.shading; sh.show_shadows=True; sh.shadow_intensity=0.30; sh.background_type='WORLD' if hasattr(sh,'background_type') else None
sc.view_settings.view_transform='Standard'
set_cam((TX,TY,float(TZ)),float(AZ),float(EL),float(DIST),lens=35)
ZIN=float(os.environ.get('ZIN','0.0'))   # >0: dynamic framing (smoothed head height) with this margin
if ZIN>0 and '--still' not in sys.argv:
    fr=list(range(sc.frame_start,sc.frame_end+1)); hz=[]
    for f in fr:
        sc.frame_set(f); bpy.context.view_layer.update(); hz.append(max(arm.pose.bones['Head'].head[2]+0.2, arm.pose.bones['Hand.L'].head[2]+0.15 if False else 0))
    hz=np.array(hz); n=len(hz); k=45; pad=np.concatenate([np.full(k,hz[0]),hz,np.full(k,hz[-1])]); sm=np.convolve(pad,np.ones(2*k+1)/(2*k+1),mode='same')[k:k+n]
    top=np.maximum(sm,float(os.environ.get('ZMIN','1.05')))+ZIN
    for f,tp in zip(fr,top):
        c=(tp/2.0); d=tp/0.5786*float(os.environ.get('ZK','1.0'))
        set_cam((TX,TY,c),float(AZ),float(EL),d,lens=35)
        for ob in (bpy.data.objects['Cam'],bpy.data.objects['CamTarget']): ob.keyframe_insert('location',frame=f)
sc.render.image_settings.file_format='PNG'
import os; os.makedirs(OUT,exist_ok=True); sc.render.filepath=OUT+'/f_'
if '--still' in sys.argv:
    sc.frame_set(int(sys.argv[sys.argv.index('--still')+1])); sc.render.filepath=OUT+'.png'; bpy.ops.render.render(write_still=True)
else: bpy.ops.render.render(animation=True)
print('RENDERED',OUT)
