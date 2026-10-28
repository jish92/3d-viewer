import bpy,sys,numpy as np,math,json
from mathutils import Matrix,Vector
from scipy.interpolate import CubicSpline
IN,OUT=sys.argv[sys.argv.index('--')+1:sys.argv.index('--')+3]
bpy.ops.wm.open_mainfile(filepath=IN)
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
bpy.context.view_layer.objects.active=arm; bpy.ops.object.mode_set(mode='POSE')
pb=arm.pose.bones
P=json.load(open('/workspace/me3d/swim_female/gait_params.json'))['bust']
F0,ZETA,PEAK=P['freq_hz'],P['zeta'],P['peak_run_m']
om=2*math.pi*F0
tracks={t.name:t for t in arm.animation_data.nla_tracks}
def only(name):
    for n,t in tracks.items(): t.mute=(n!=name)
def series(name):
    only(name); st=tracks[name].strips[0]; f0,f1=int(st.frame_start),int(round(st.frame_end))
    dup = name in ('Walk','Run')            # last key duplicates the first one
    frames=list(range(f0,f1+1-(1 if dup else 0)))
    out={}
    for sd in 'LR':
        z=[];Rpar=[]
        for f in frames:
            bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
            z.append(pb['Bust.'+sd].head[2]); Rpar.append(np.array(pb['Chest'].matrix.to_3x3()))
        out[sd]=(np.array(z),np.array(Rpar))
    return frames,out
def spring(z):
    n=len(z); zz=np.concatenate([z,z,z,z,z]); t=np.arange(len(zz))/30.0
    cs=CubicSpline(t,zz,bc_type='natural'); 
    sub=4; dt=1/30/sub; d=0.0; v=0.0; res=[]
    ts=np.arange(n*2*30*sub//1)*0  # unused
    T0=n/30.0*1.0  # skip two periods warm-up
    tt=np.arange(0,5*n/30.0-1e-9,dt)
    dd=np.zeros(len(tt))
    for i,tm in enumerate(tt):
        a=float(cs(tm,2))
        acc=-a-2*ZETA*om*v-om*om*d
        v+=acc*dt; d+=v*dt; dd[i]=d
    k0=3*n*sub        # last-but-one period (settled)
    return dd[k0:k0+n*sub:sub]
res={}
for name in ('Walk','Run','MotionBricks'):
    frames,s=series(name); res[name]=(frames,{sd:spring(s[sd][0]) for sd in 'LR'},s)
rawpeak={n:max(np.abs(r[1]['L']).max(),np.abs(r[1]['R']).max()) for n,r in res.items()}
g_run=min(1.0,PEAK/rawpeak['Run'])
gain={'Run':g_run,'MotionBricks':min(g_run,PEAK/rawpeak['MotionBricks']),'Walk':min(P['walk_rel']*g_run,P['peak_walk_m']/rawpeak['Walk'])}
print('raw (unscaled) spring peaks m:',{k:round(v,4) for k,v in rawpeak.items()}); print('gains',{k:round(v,3) for k,v in gain.items()})
bpy.ops.object.mode_set(mode='POSE')
Rb=np.array(pb['Bust.L'].bone.matrix_local.to_3x3()); 
report={}
for name in ('Walk','Run','MotionBricks'):
    frames,dd,s=res[name]; only(name)
    act=bpy.data.actions[name]; 
    arm.animation_data.action=act
    for t in tracks.values(): t.mute=True
    pk=0
    for sd in 'LR':
        b=pb['Bust.'+sd]; Rbb=np.array(b.bone.matrix_local.to_3x3()); Rbc=np.array(pb['Chest'].bone.matrix_local.to_3x3())
        for i,f in enumerate(frames+([frames[0]+len(frames)] if name in('Walk','Run') else [])):
            ii=i%len(frames); d=dd[sd][ii]*gain[name]; pk=max(pk,abs(d))
            Rp=s[sd][1][ii]
            loc=np.linalg.inv(Rbb)@Rbc@np.linalg.inv(Rp)@np.array([0,0,d])
            b.location=Vector(loc); b.keyframe_insert('location',frame=f)
        b.location=(0,0,0)
    report[name]=pk; print(name,'bust displacement peak mm',round(pk*1000,2))
arm.animation_data.action=None
for t in tracks.values(): t.mute=False

# ---------------- glute secondary motion (children of Hips) ----------------
G=json.load(open('/workspace/me3d/swim_female/gait_params.json'))['glute']
omg=2*math.pi*G['freq_hz']; ZG=G['zeta']
def gseries(name):
    only(name); st=tracks[name].strips[0]; f0,f1=int(st.frame_start),int(round(st.frame_end))
    dup = name in ('Walk','Run'); frames=list(range(f0,f1+1-(1 if dup else 0)))
    out={sd:{'P':[],'H':[],'Rp':[]} for sd in 'LR'}
    for f in frames:
        bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
        for sd in 'LR':
            out[sd]['P'].append(np.array(pb['Glute.'+sd].head)); out[sd]['H'].append(np.array(pb['Hips'].head)); out[sd]['Rp'].append(np.array(pb['Hips'].matrix.to_3x3()))
    return frames,{sd:{k:np.array(v) for k,v in o.items()} for sd,o in out.items()}
def gspring(x):          # x: (n,) input coordinate series -> damped spring displacement driven by -accel
    n=len(x); zz=np.concatenate([x]*5); t=np.arange(len(zz))/30.0
    cs=CubicSpline(t,zz,bc_type='natural'); sub=4; dt=1/30/sub; d=0.0; v=0.0
    tt=np.arange(0,5*n/30.0-1e-9,dt); dd=np.zeros(len(tt))
    for i,tm in enumerate(tt):
        a=float(cs(tm,2)); acc=-a-2*ZG*omg*v-omg*omg*d; v+=acc*dt; d+=v*dt; dd[i]=d
    k0=3*n*sub; return dd[k0:k0+n*sub:sub]
gres={}
for name in ('Walk','Run','MotionBricks'):
    frames,s=gseries(name); disp={}
    for sd in 'LR':
        P=s[sd]['P']; r=P-s[sd]['H']
        dz=gspring(P[:,2])                                   # pelvis vertical acceleration (+ the point's own rise/fall)
        dh=np.stack([gspring(r[:,0]),gspring(r[:,1])],1)     # acceleration of the point about the pelvis centre = response to yaw / tilt
        disp[sd]=(dz,dh)
    pv=max(np.abs(disp[sd][0]).max() for sd in 'LR'); ph=max(np.linalg.norm(disp[sd][1],axis=1).max() for sd in 'LR')
    tgt={'Walk':G['peak_walk_m'],'Run':G['peak_run_m'],'MotionBricks':G['peak_mb_m']}[name]
    gv=tgt/pv; gh=G['horiz_rel']*tgt/max(ph,1e-9)
    gres[name]=(frames,s,disp,gv,gh); print('glute',name,'raw vertical peak mm',round(pv*1000,2),'raw horiz peak mm',round(ph*1000,2),'gains',round(gv,3),round(gh,3))
grep={}
Rbh=np.array(pb['Hips'].bone.matrix_local.to_3x3())
for name in ('Walk','Run','MotionBricks'):
    frames,s,disp,gv,gh=gres[name]; only(name)
    arm.animation_data.action=bpy.data.actions[name]
    for t in tracks.values(): t.mute=True
    pk=0
    for sd in 'LR':
        b=pb['Glute.'+sd]; Rbb=np.array(b.bone.matrix_local.to_3x3()); dz,dh=disp[sd]
        for i,f in enumerate(frames+([frames[0]+len(frames)] if name in('Walk','Run') else [])):
            ii=i%len(frames); dw=np.array([dh[ii,0]*gh,dh[ii,1]*gh,dz[ii]*gv]); pk=max(pk,np.linalg.norm(dw))
            loc=np.linalg.inv(Rbb)@Rbh@np.linalg.inv(s[sd]['Rp'][ii])@dw
            b.location=Vector(loc); b.keyframe_insert('location',frame=f)
        b.location=(0,0,0)
    grep[name]=pk; print(name,'glute displacement peak mm',round(pk*1000,2))
arm.animation_data.action=None
for t in tracks.values(): t.mute=False
json.dump(grep,open('/workspace/me3d/swim_female/out/glute_report.json','w'))
bpy.ops.object.mode_set(mode='OBJECT')
json.dump(report,open('/workspace/me3d/swim_female/out/bust_report.json','w'))
bpy.ops.wm.save_as_mainfile(filepath=OUT); print('SAVED')
