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
add_leg_helpers(arm,body)
from scipy.spatial.transform import Rotation as _R, Slerp as _S
def _mid(A,B): return _S([0,1],_R.from_matrix([A,B]))([0.5]).as_matrix()[0]
def basis_h(D,pelvis):
    bs,loc=rig.basis(D,pelvis); Db={n:C@D[n]@C.T for n in D}
    for sd in 'LR':
        for nm,par,ch in (('UpperLeg_Knee.'+sd,'UpperLeg.'+sd,'LowerLeg.'+sd),('UpperLeg_Hip.'+sd,'Hips','UpperLeg.'+sd)):
            Rcr=rig.Rrest_b[par]; Dm=_mid(Db[par],Db[ch]); bs[nm]=Rcr.T@Db[par].T@Dm@Rcr
    return bs,loc
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
import yk
KO='/workspace/me3d/kim_out/'
KIMO={'ksquat':('Yoga_DeepSquat',KO+'squat.npz',lambda:yk.squat_arms(rig,rest)),
 'kwarrior2':('Yoga_WarriorII',KO+'warrior2.npz',lambda:None),'kdowndog':('Yoga_DownwardDog',KO+'downdogb.npz',lambda:None),
 'kpuppy':('Yoga_Puppy',KO+'puppyb.npz',lambda:None),'khappybaby':('Yoga_HappyBaby',KO+'happybabyb.npz',lambda:None),
 'ktree':('Yoga_Tree',KO+'treeb.npz',lambda:None),'kchild':('Yoga_Child',KO+'childb.npz',lambda:None)}
if os.environ.get('SMILE'):
    add_smile(body); body.data.shape_keys.key_blocks['Smile'].value=1.0
if os.environ.get('PURGE'):
    for a in list(bpy.data.actions): bpy.data.actions.remove(a)
    for t in list(arm.animation_data.nla_tracks): arm.animation_data.nla_tracks.remove(t)
def floor_pass(clip,iters=2):
    # lift/lower whole body per frame so the lowest mesh vertex touches z=0 (keeps feet planted on the floor)
    N=clip.frames(); dz=np.zeros(N)
    for it in range(iters):
        mz=np.zeros(N)
        for f in range(N):
            P=clip.Ps[f]; D,pos,info=rig.solve(P); bs,loc=basis_h(D,P['pelvis'])
            for n,Rm in bs.items():
                pb=arm.pose.bones[n]; pb.rotation_mode='QUATERNION'; pb.rotation_quaternion=Matrix(Rm.tolist()).to_quaternion()
                if n=='Hips': pb.location=Vector(loc)
            co,_=evalmesh(); mz[f]=co[:,2].min()
        k=3; pad=np.concatenate([np.repeat(mz[:1],k),mz,np.repeat(mz[-1:],k)]); mzs=np.convolve(pad,np.ones(2*k+1)/(2*k+1),'valid')
        d=-np.minimum(mzs,mz)+0*mz if False else -(mzs)
        print('floor pass',it,'minz range',mz.min(),mz.max())
        for f in range(N):
            clip.Ps[f]['pelvis']=clip.Ps[f]['pelvis']+np.array([0,d[f],0])
            for sd in 'LR': clip.Ps[f]['leg'+sd]['ankle']=clip.Ps[f]['leg'+sd]['ankle']+np.array([0,d[f],0])
def setpose(P):
    D,pos,info=rig.solve(P); bs,loc=basis_h(D,P['pelvis'])
    for n,Rm in bs.items():
        pb=arm.pose.bones[n]; pb.rotation_mode='QUATERNION'; pb.rotation_quaternion=Matrix(Rm.tolist()).to_quaternion()
        if n=='Hips': pb.location=Vector(loc)
def arm_pass(clip,thr=0.010,iters=4,step=0.012):
    N=clip.frames(); tg=np.concatenate([bodyonly,legL,legR]); off=np.zeros((N,2,3)); step_v=[np.array([1,0,0.]),np.array([0,0,1.]),np.array([0.7,0.3,0.7])]
    def apply(f,o):
        P=copy.deepcopy(clip.Ps[f])
        for i,sd in enumerate('LR'):
            sg=1 if sd=='L' else -1; a=P['arm'+sd]; a['wrist']=a['wrist']+o[i]*np.array([sg,1,1.]); a['pole']=a['pole']+0.0
        return P
    def pen_f(f,o):
        setpose(apply(f,o)); co,nn=evalmesh(); return pen(co,nn,armtest,tg)
    import copy
    for it in range(iters):
        bad=0
        for f in range(N):
            p0=pen_f(f,off[f])
            if p0<=thr: continue
            bad+=1; best=(p0,None)
            for v in step_v:
                o=off[f].copy(); o[0]=o[0]+v*step; o[1]=o[1]+v*step
                p=pen_f(f,o)
                if p<best[0]: best=(p,o)
            if best[1] is not None: off[f]=best[1]
        # smooth offsets in time
        k=4; pad=np.concatenate([np.repeat(off[:1],k,0),off,np.repeat(off[-1:],k,0)]); off=np.stack([np.convolve(pad[:,i,j],np.ones(2*k+1)/(2*k+1),'valid') for i in range(2) for j in range(3)],1).reshape(N,2,3)
        print('arm pass',it,'bad frames',bad)
        if bad==0: break
    for f in range(N): clip.Ps[f]=apply(f,off[f])
LIFT=None
def foot_pass(clip,iters=2,step=3):
    global LIFT
    N=clip.frames(); LIFT={sd:np.zeros(N) for sd in 'LR'}
    for it in range(iters):
        fr=list(range(0,N,step))+[N-1]; need={sd:[] for sd in 'LR'}
        for f in fr:
            P=getpl(clip,f); setpose(P); co,_=evalmesh()
            for sd in 'LR': need[sd].append(max(0.0,-float(co[GS['Foot.'+sd],2].min())))
        for sd in 'LR':
            n=np.interp(np.arange(N),fr,need[sd]); k=2; pad=np.concatenate([np.repeat(n[:1],k),n,np.repeat(n[-1:],k)]); n=np.maximum(np.convolve(pad,np.ones(2*k+1)/(2*k+1),'valid'),n*0.0)
            n=np.maximum(n,np.interp(np.arange(N),fr,need[sd])*0.9)
            LIFT[sd]=LIFT[sd]+n
        print('foot pass',it,{sd:round(float(max(need[sd])),4) for sd in 'LR'})
def getpl(clip,f):
    import copy
    P=copy.deepcopy(clip.P(f/yclips.FPS))
    if LIFT is not None:
        for sd in 'LR': P['leg'+sd]['ankle']=np.asarray(P['leg'+sd]['ankle'],float)+np.array([0,LIFT[sd][f],0])
    return P
for key in WANT:
    if key in KIMO:
        nm,npz,bf=KIMO[key]; clip,kinfo=yk.make(nm,npz,rig,rest,blend_fn=bf()); print('kimo',kinfo['k'],kinfo['ka'],clip.hold); floor_pass(clip)
        if os.environ.get('ARMPASS'): arm_pass(clip)
    else:
        clip=getattr(yclipdefs,'clip_'+key)(rig,rest)
        LIFT=None
        if os.environ.get('FOOTPASS'): foot_pass(clip)
    N=clip.frames(); name=clip.name
    act=bpy.data.actions.new(name); arm.animation_data.action=act
    for t in arm.animation_data.nla_tracks: t.mute=True
    pel=[];chest=[];hipsD=[];chestD=[];info_all=[]; prevq={}
    for f in range(N):
        P=getpl(clip,f)
        D,pos,info=rig.solve(P)
        bs,loc=basis_h(D,P['pelvis'])
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
