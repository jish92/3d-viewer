# usage: blender --python build_yoga.py -- out.blend [clip names...]
import bpy,sys,os,json,numpy as np,math
sys.path.insert(0,'/workspace/me3d/swim_female/yoga')
exec(open('/workspace/me3d/swim_female/yoga/ybl.py').read())
import yclips,yclipdefs,yposes
from scipy.spatial import cKDTree
args=sys.argv[sys.argv.index('--')+1:]; OUT=args[0]; WANT=args[1:] or ['squat','warrior','tree','dog','puppy']
bpy.ops.wm.open_mainfile(filepath='/workspace/me3d/swim_female/out/her_final.blend')
arm=get_arm(); body=get_body(); rig=build_rig(arm); rest=rest_params(rig)
bpy.context.view_layer.objects.active=arm; bpy.ops.object.mode_set(mode='POSE'); clear_nla(arm)
for t in arm.animation_data.nla_tracks: t.mute=False
smooth_hip_weights(body,iters=10)
G=json.load(open('/workspace/me3d/swim_female/gait_params.json'))
GS=group_sets(body)
nv=len(body.data.vertices); gname={g.index:g.name for g in body.vertex_groups}
dom=np.full(nv,-1); wm=np.zeros(nv)
for v in body.data.vertices:
    for g in v.groups:
        if g.weight>wm[v.index]: wm[v.index]=g.weight; dom[v.index]=g.group
dn=np.array([gname.get(d,'') for d in dom])
restco=mesh_co_rest=np.array([v.co[:] for v in body.data.vertices])
shL=rig.head['UpperArm.L']; shR=rig.head['UpperArm.R']
Rb=lambda p:np.array([p[0],-p[2],p[1]])   # g -> blender
def idx(f): return np.where(np.array([f(n) for n in dn]))[0]
armv=idx(lambda n:n.startswith(('LowerArm','Hand','UpperArm')))
fore=idx(lambda n:n.startswith(('LowerArm','Hand')))
upa=idx(lambda n:n.startswith('UpperArm')); dsh=np.minimum(np.linalg.norm(restco[upa]-np.array(arm.data.bones['UpperArm.L'].head_local),axis=1),np.linalg.norm(restco[upa]-np.array(arm.data.bones['UpperArm.R'].head_local),axis=1))
armtest=np.concatenate([fore,upa[dsh>0.10]])
torso=idx(lambda n:n.startswith(('Hips','Spine','Chest','Bust','Glute','UpperLeg','LowerLeg','Foot')))
headv=idx(lambda n:n in ('Head','Neck'))
legL=idx(lambda n:n.endswith('.L') and n.startswith(('UpperLeg','LowerLeg','Foot'))); legR=idx(lambda n:n.endswith('.R') and n.startswith(('UpperLeg','LowerLeg','Foot')))
shin=lambda sd:idx(lambda n:n.endswith('.'+sd) and n.startswith(('LowerLeg','Foot')))
bodyonly=idx(lambda n:n.startswith(('Hips','Spine','Chest','Bust','Glute')))
def evalmesh():
    bpy.context.view_layer.update()
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh()
    co=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',co); nn=np.empty(len(m.vertices)*3); m.vertices.foreach_get('normal',nn)
    e.to_mesh_clear(); return co.reshape(-1,3),nn.reshape(-1,3)
LAST={}
def pen(co,nn,test,tgt,rad=0.06):
    kd=cKDTree(co[tgt]); P=co[test]; d,i=kd.query(P); q=tgt[i]; sd=((P-co[q])*nn[q]).sum(1)
    p=np.where((sd<0)&(d<rad),-sd,0); k=int(np.argmax(p)); LAST['v']=gname.get(dom[test[k]],'')+'>'+gname.get(dom[q[k]],''); return float(p.max())
def spring(z,F0,Z,dt=1/30.,sub=4):
    n=len(z); zz=np.tile(z,5); t=np.arange(len(zz))*dt
    a=np.gradient(np.gradient(zz,dt),dt); om=2*math.pi*F0; d=0;v=0; res=[]
    for i in range(len(zz)):
        for _ in range(sub):
            acc=-a[i]-2*Z*om*v-om*om*d; v+=acc*dt/sub; d+=v*dt/sub
        res.append(d)
    return np.array(res[3*n:4*n])
report={}
for key in WANT:
    clip=getattr(yclipdefs,'clip_'+key)(rig,rest)
    N=clip.frames(); name=clip.name
    act=bpy.data.actions.new(name); arm.animation_data.action=act
    for t in arm.animation_data.nla_tracks: t.mute=True
    pel=[];chest=[];hipsD=[];chestD=[];info_all=[]; prevq={}
    for f in range(N):
        P=clip.P(f/yclips.FPS)
        D,pos,info=rig.solve(P)
        bs,loc=rig.basis(D,P['pelvis'])
        for n,Rm in bs.items():
            pb=arm.pose.bones[n]; pb.rotation_mode='QUATERNION'; q=Matrix(Rm.tolist()).to_quaternion()
            if n in prevq: q.make_compatible(prevq[n])
            prevq[n]=q.copy(); pb.rotation_quaternion=q; pb.keyframe_insert('rotation_quaternion',frame=f+1)
            if n=='Hips': pb.location=Vector(loc); pb.keyframe_insert('location',frame=f+1)
        info_all.append({k:dict(short=float(v['short'])) for k,v in info.items()})
        dBust=pos['Chest']+D['Chest']@(rig.head['Bust.L']-rig.head['Chest']); dGl=pos['Hips']+D['Hips']@(rig.head['Glute.L']-rig.head['Hips'])
        pel.append(dGl[1]); chest.append(dBust[1]); hipsD.append(D['Hips']); chestD.append(D['Chest'])
        yclips_last=(P,pos,D)
    # springs (subtle)
    for nm,series,Dl,par,pk in (('Bust',chest,chestD,'Chest',G['bust']),('Glute',pel,hipsD,'Hips',G['glute'])):
        d=spring(np.array(series),pk['freq_hz'],pk['zeta']); raw=float(np.abs(d).max()); cap=0.0035
        gain=min(1.0,cap/max(raw,1e-9))
        for sd in 'LR':
            b=arm.pose.bones[nm+'.'+sd]; Rrb=rig.Rrest_b[nm+'.'+sd]
            for f in range(N):
                Db=C@Dl[f]@C.T; loc=Rrb.T@Db.T@np.array([0,0,d[f]*gain])
                b.location=Vector(loc); b.keyframe_insert('location',frame=f+1)
            b.location=(0,0,0)
        print(name,nm,'raw spring peak mm %.2f gain %.2f'%(raw*1000,gain))
    # QA over frames (every 3rd)
    worst={}; stats=dict(minz=9,short=max(max(v['short'] for v in i.values()) for i in info_all))
    for f in range(0,N,3):
        bpy.context.scene.frame_set(f+1); co,nn=evalmesh()
        stats['minz']=min(stats['minz'],float(co[:,2].min()))
        for kk,(ta,tb) in dict(arm_pen=(armtest,np.concatenate([bodyonly,legL,legR])),head_pen=(armtest,headv),leg_pen=(legL,legR),leg_pen2=(legR,legL),torso_pen=(np.concatenate([shin('L'),shin('R')]),bodyonly)).items():
            p=pen(co,nn,ta,tb)
            if p>stats.get(kk,0): stats[kk]=p; worst[kk]=(f,LAST['v'])
    print(name,'frames',N,{k:round(v,4) for k,v in stats.items()},worst); report[name]=stats
    arm.animation_data.action=None
    tr=arm.animation_data.nla_tracks.new(); tr.name=name; st=tr.strips.new(name,1,act); st.name=name
    for t in arm.animation_data.nla_tracks: t.mute=False
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=OUT)
json.dump(report,open(OUT+'.report.json','w'),indent=1); print('SAVED',OUT)
