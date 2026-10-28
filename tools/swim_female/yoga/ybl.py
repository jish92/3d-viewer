# blender-side helpers: build Rig, apply pose params, mesh evaluation
import bpy,sys,numpy as np,math
sys.path.insert(0,'/workspace/me3d/swim_female/yoga')
import importlib,ylib; importlib.reload(ylib)
from ylib import *
from mathutils import Matrix,Quaternion,Vector
def get_arm(): return [o for o in bpy.data.objects if o.type=='ARMATURE'][0]
def get_body(): return [o for o in bpy.data.objects if o.type=='MESH'][0]
def build_rig(arm):
    B={}
    for b in arm.data.bones:
        B[b.name]=dict(head=np.array(b.head_local),tail=np.array(b.tail_local),parent=b.parent.name if b.parent else None,R=np.array(b.matrix_local.to_3x3()))
    return Rig(B)
def rest_params(rig):
    P=dict(pelvis=rig.head['Hips'].copy())
    for sd in 'LR':
        ft=rig.head['Foot.'+sd]; P['leg'+sd]=dict(ankle=ft.copy(),pole=np.array([0,0,1.]),yaw=0,pitch=0,roll=0)
        sh=rig.head['UpperArm.'+sd]; el=rig.head['LowerArm.'+sd]; wr=rig.head['Hand.'+sd]
        u=nrm(wr-sh); pole=-(el-sh-u*((el-sh)@u))
        P['arm'+sd]=dict(mode='world',wrist=wr.copy(),pole=nrm(pole)*-1,hand=rig.d0['Hand.'+sd].copy(),palm=rig.s0['Hand.'+sd].copy())
    return P
def apply_pose(arm,rig,P,frame=None):
    D,pos,info=rig.solve(P)
    bs,loc=rig.basis(D,P['pelvis'])
    for n,Rb in bs.items():
        pb=arm.pose.bones[n]; pb.rotation_mode='QUATERNION'
        q=Matrix(Rb.tolist()).to_quaternion(); pb.rotation_quaternion=q
        if n=='Hips': pb.location=Vector(loc)
        if frame is not None:
            pb.keyframe_insert('rotation_quaternion',frame=frame)
            if n=='Hips': pb.keyframe_insert('location',frame=frame)
    return D,pos,info
def mesh_co(body):
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh(); co=np.empty(len(m.vertices)*3); m.vertices.foreach_get('co',co); co=co.reshape(-1,3).copy(); e.to_mesh_clear(); return co
def clear_nla(arm):
    for t in arm.animation_data.nla_tracks: t.mute=True
    arm.animation_data.action=None

def setup_scene(engine='BLENDER_WORKBENCH',res=(1280,720),fps=24,floor=True):
    sc=bpy.context.scene; sc.render.engine=engine; sc.render.resolution_x,sc.render.resolution_y=res; sc.render.resolution_percentage=100; sc.render.fps=fps
    w=bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new('W'); sc.world=w; w.use_nodes=True
    bg=w.node_tree.nodes['Background']; bg.inputs[0].default_value=(0.80,0.80,0.81,1); bg.inputs[1].default_value=1.0
    if engine=='BLENDER_WORKBENCH':
        sh=sc.display.shading; sh.light='STUDIO'; sh.color_type='VERTEX'; sh.show_shadows=True; sh.shadow_intensity=0.35; sh.show_cavity=False
        sh.show_object_outline=False; sh.studio_light='rim.sl' if False else sh.studio_light
        sc.view_settings.view_transform='Standard'; sc.render.film_transparent=False
        sc.display.render_aa='8'
    if floor and 'Floor' not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=30,location=(0,0,-0.0005)); f=bpy.context.active_object; f.name='Floor'
        m=bpy.data.materials.new('FloorM'); m.diffuse_color=(0.80,0.80,0.81,1); f.data.materials.append(m)
    if 'Cam' not in bpy.data.objects:
        cd=bpy.data.cameras.new('Cam'); co=bpy.data.objects.new('Cam',cd); sc.collection.objects.link(co); sc.camera=co
        tg=bpy.data.objects.new('CamTarget',None); sc.collection.objects.link(tg)
        c=co.constraints.new('TRACK_TO'); c.target=tg; c.track_axis='TRACK_NEGATIVE_Z'; c.up_axis='UP_Y'
def set_cam(target,az,el,dist,lens=50,shift=(0,0)):
    co=bpy.data.objects['Cam']; tg=bpy.data.objects['CamTarget']; tg.location=target
    a=math.radians(az); e=math.radians(el)
    co.location=(target[0]+dist*math.sin(a)*math.cos(e),target[1]-dist*math.cos(a)*math.cos(e),target[2]+dist*math.sin(e))
    co.data.lens=lens; co.data.shift_x,co.data.shift_y=shift

def rear_hip_fix(body,frac=0.6,y0=0.66,y1=0.90,zr0=0.0,zr1=0.05):
    """shift part of the rear/top-thigh weight to the pelvis so deep hip flexion does not fold the thigh flesh into the buttock (blender coords: z up, +y = back)"""
    n=len(body.data.vertices); vg={g.name:g for g in body.vertex_groups}
    V=np.array([v.co[:] for v in body.data.vertices])
    sm=lambda t:(lambda u:u*u*(3-2*u))(np.clip(t,0,1))
    f=frac*sm((V[:,2]-y0)/(y1-y0))*sm((V[:,1]-zr0)/(zr1-zr0))
    ih=vg['Hips']
    for sd in 'LR':
        g=vg['UpperLeg.'+sd]
        for v in body.data.vertices:
            w=0.0
            for e in v.groups:
                if e.group==g.index: w=e.weight
            if w>0 and f[v.index]>0:
                mv=w*f[v.index]; g.add([v.index],w-mv,'REPLACE'); ih.add([v.index],mv,'ADD')

def smooth_hip_weights(body,iters=10,y0=0.70,y1=0.97,lam=0.5):
    """smooth the hip/thigh skin weights (core bones only; Bust/Glute spring weights untouched); the suit copies the skin weights beneath it"""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    me=body.data; n=len(me.vertices); names=[g.name for g in body.vertex_groups]; gidx={g.index:i for i,g in enumerate(body.vertex_groups)}
    W=np.zeros((n,len(names)))
    for v in me.vertices:
        for e in v.groups: W[v.index,gidx[e.group]]=e.weight
    spring=np.array([nm.startswith(('Bust','Glute')) for nm in names])
    col=np.empty(n*4); me.color_attributes['Col'].data.foreach_get('color',col); col=col.reshape(-1,4)[:,:3]
    V=np.array([v.co[:] for v in me.vertices]); z=V[:,2]
    skin=(col[:,0]>0.45)&(col[:,0]>col[:,2]*1.5); suit=(col[:,0]<0.05)&(col[:,2]>0.02)&(col[:,2]>col[:,0]*2.5)
    E=np.empty(len(me.edges)*2,int); me.edges.foreach_get('vertices',E); E=E.reshape(-1,2)
    keep=skin[E[:,0]]&skin[E[:,1]]; E=E[keep]
    A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(n,n)); A=(A+A.T).tocsr(); A.data[:]=1; deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
    band=np.clip(np.minimum((z-y0)/0.04,(y1-z)/0.04),0,1)*(np.abs(V[:,0])<0.25)
    bs=(band*skin)[:,None]
    Wc=W.copy(); Wc[:,spring]=0; room=1-W[:,spring].sum(1,keepdims=True)
    tot=np.maximum(Wc.sum(1,keepdims=True),1e-9); Wc/=tot            # normalised core weights
    for _ in range(iters):
        Wn=(A@Wc)/deg[:,None]; Wc=Wc*(1-lam*bs)+Wn*(lam*bs)
    ii=np.argsort(-Wc,1)[:,4:]; np.put_along_axis(Wc,ii,0,1); Wc/=np.maximum(Wc.sum(1,keepdims=True),1e-9)
    Wc*=room if True else 1
    Wn_=W.copy(); m=(bs[:,0]>0)
    Wn_[m]=np.where(spring[None,:],W[m],Wc[m])
    # suit: copy from the nearest smoothed skin vertices (rest positions), keep its own spring weights
    sk=np.where(skin)[0]; tr=cKDTree(V[sk]); su=np.where(suit&(band>0))[0]
    d,ik=tr.query(V[su],k=4); wk=1/(d+2e-3); wk/=wk.sum(1,keepdims=True)
    src=Wn_[sk[ik]]; Wsu=(src*wk[...,None]).sum(1); 
    Wsu[:,spring]=W[su][:,spring]; core=~spring; sc=(1-Wsu[:,spring].sum(1))/np.maximum(Wsu[:,core].sum(1),1e-9); Wsu[:,core]*=sc[:,None]
    b=band[su][:,None]; Wn_[su]=W[su]*(1-b)+Wsu*b
    for g in body.vertex_groups: g.remove(list(range(n)))
    for j,g in enumerate(body.vertex_groups):
        for v in np.where(Wn_[:,j]>1e-4)[0]: g.add([int(v)],float(Wn_[v,j]),'REPLACE')

def group_sets(body,thr=0.5):
    names={g.index:g.name for g in body.vertex_groups}; S={}
    for v in body.data.vertices:
        for e in v.groups:
            if e.weight>=thr: S.setdefault(names[e.group],[]).append(v.index)
    return {k:np.array(v) for k,v in S.items()}
def floor_report(body,V,GS):
    out={'all':float(V[:,2].min())}
    for g in ('Hand.L','Hand.R','Foot.L','Foot.R','LowerLeg.L','LowerArm.L','Head','Chest','Hips'):
        if g in GS: out[g]=round(float(V[GS[g],2].min()),3)
    return out

def add_smile(body,name='Smile',amt=1.0,MZ=1.557,MLIFT=0.0065,CLIFT=0.0040):
    """subtle relaxed closed-mouth smile: mouth-corner lift, cheek lift, softened (slightly lifted lower / lowered upper) lids"""
    me=body.data; V=np.array([v.co[:] for v in me.vertices]); n=len(V)
    col=np.empty(n*4); me.color_attributes['Col'].data.foreach_get('color',col); col=col.reshape(-1,4)[:,:3]
    ratio=col[:,1]/np.maximum(col[:,0],1e-3)
    skin=(col[:,0]>0.55)&(ratio<0.7)&(V[:,1]<-0.04)&(V[:,2]>1.52)&(V[:,2]<1.70)&(np.abs(V[:,0])<0.10)
    if me.shape_keys is None: body.shape_key_add(name='Basis',from_mix=False)
    D=np.zeros((n,3)); x=V[:,0]; ax=np.abs(x); sx=np.sign(x); z=V[:,2]
    g=lambda d,s: np.exp(-(d/s)**2)
    # mouth: corners up and slightly out/back, parabola along the lip line
    fx=np.where(ax<=0.024,(ax/0.024)**2,g(ax-0.024,0.016)); fz=g(z-MZ,0.013)
    w=fx*fz*skin
    D[:,2]+=MLIFT*w; D[:,0]+=sx*0.0020*w; D[:,1]+=0.0015*w
    # cheeks lift & fullness
    wc=g(np.hypot(ax-0.050,z-1.572),0.022)*skin
    D[:,2]+=CLIFT*wc; D[:,1]-=0.0025*wc; D[:,0]+=sx*0.0006*wc
    # eyes: find eye centres from the eyeballs (ivory verts), soften lids
    iv=(ratio>0.9)&(col[:,0]>0.7)&(z>1.62)&(V[:,1]<-0.03)&(ax<0.07)
    ez=float(np.median(z[iv])) if iv.sum()>10 else 1.665
    exs=ax[iv]; ex=float(np.median(exs)) if iv.sum()>10 else 0.033
    rr=0.0125; ez=1.620; ex=0.030
    low=g(np.hypot((ax-ex)*0.7,z-(ez-rr)),0.010)*skin; up=g(np.hypot((ax-ex)*0.7,z-(ez+rr)),0.010)*skin
    D[:,2]+=0.0014*low-0.0028*up; D[:,1]-=0.0010*up
    kb=me.shape_keys.key_blocks.get(name) or body.shape_key_add(name=name,from_mix=False)
    co=(V+D*amt).reshape(-1); kb.data.foreach_set('co',co); kb.value=0.0
    kb.slider_max=1.5
    return kb,dict(eye_z=ez,eye_x=ex,moved=int((np.abs(D).sum(1)>1e-5).sum()),max_mm=float(np.abs(D).max()*1000))

def add_leg_helpers(arm,body,kb=0.5,hb=0.5,ks=0.075,hs=0.07):
    """extra (rotation-bisecting) helper bones at knees and hips: UpperLeg_Knee.S (child of UpperLeg, pivot at knee) and UpperLeg_Hip.S (child of Hips, pivot at hip); weights taken from the neighbouring bones"""
    import numpy as np
    bpy.context.view_layer.objects.active=arm; bpy.ops.object.mode_set(mode='EDIT'); eb=arm.data.edit_bones
    for sd in 'LR':
        for nm,par,hd in (('UpperLeg_Knee.'+sd,'UpperLeg.'+sd,eb['LowerLeg.'+sd].head.copy()),('UpperLeg_Hip.'+sd,'Hips',eb['UpperLeg.'+sd].head.copy())):
            b=eb.new(nm); P=eb[par]; d=(P.tail-P.head).normalized()
            b.head=hd; b.tail=hd+d*0.06; b.roll=P.roll; b.parent=P; b.use_connect=False; b.use_deform=True
    bpy.ops.object.mode_set(mode='POSE')
    me=body.data; n=len(me.vertices); V=np.array([v.co[:] for v in me.vertices]); z=V[:,2]; x=V[:,0]
    names=[g.name for g in body.vertex_groups]; gi={g.index:i for i,g in enumerate(body.vertex_groups)}
    W=np.zeros((n,len(names)+4))
    for v in me.vertices:
        for e in v.groups: W[v.index,gi[e.group]]=e.weight
    ix=lambda s:names.index(s)
    new=[]
    for sd in 'LR':
        ul,ll,hp=ix('UpperLeg.'+sd),ix('LowerLeg.'+sd),ix('Hips'); ulo=ix('UpperLeg.'+('R' if sd=='L' else 'L'))
        sg=1 if sd=='L' else -1
        # knee: take a bell-shaped share of UL+LL mass
        kz=arm.data.bones['LowerLeg.'+sd].head_local[2]; hz=arm.data.bones['UpperLeg.'+sd].head_local[2]
        bell=np.exp(-((z-kz)/ks)**2)*(x*sg>0.02)
        m=W[:,ul]+W[:,ll]; f=kb*bell; h=f*m
        s=np.maximum(m,1e-9); W[:,ul]-=h*W[:,ul]/s; W[:,ll]-=h*W[:,ll]/s; kcol=h
        # hip: share of this-side UL and the sided part of Hips, near the hip joint
        side=np.clip(0.5+x*sg/0.06,0,1); bellh=np.exp(-((z-hz)/hs)**2)
        mh=W[:,ul]+W[:,hp]*side; hh=hb*bellh*mh; s2=np.maximum(mh,1e-9)
        a=hh*W[:,ul]/s2; b=hh*W[:,hp]*side/s2
        W[:,ul]-=a; W[:,hp]-=b
        new+=[('UpperLeg_Knee.'+sd,kcol),('UpperLeg_Hip.'+sd,a+b)]
    for nm,col in new:
        g=body.vertex_groups.new(name=nm)
        for v in np.where(col>1e-4)[0]: g.add([int(v)],float(col[v]),'REPLACE')
    for j,nm in enumerate(names):
        g=body.vertex_groups[nm]
        for v in np.where(W[:,j]>1e-4)[0]: g.add([int(v)],float(W[v,j]),'REPLACE')
        for v in np.where((W[:,j]<=1e-4))[0]:
            pass
    # clear stale tiny weights
    for j,nm in enumerate(names):
        g=body.vertex_groups[nm]; zero=np.where(W[:,j]<=1e-4)[0]
        if len(zero): g.remove([int(v) for v in zero])
