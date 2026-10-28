# yoga pose library: analytic IK + world-space bone deltas for the swimher rig (all maths in glTF-like frame: x=her left, y=up, z=front)
import numpy as np, math
C=np.array([[1,0,0],[0,0,-1],[0,1,0.]])        # g -> blender
def nrm(v): v=np.asarray(v,float); return v/max(np.linalg.norm(v),1e-12)
def Rx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Eul(yaw=0,pitch=0,roll=0): return Ry(math.radians(yaw))@Rx(math.radians(pitch))@Rz(math.radians(roll))
def frame(d,s):
    d=nrm(d); s=np.asarray(s,float); s=s-d*(s@d)
    if np.linalg.norm(s)<1e-6: s=np.array([1.,0,0])-d*d[0]
    s=nrm(s); return np.stack([d,s,np.cross(d,s)],1)
def aimR(d0,s0,d1,s1): return frame(d1,s1)@frame(d0,s0).T
def ik2(H,A,L1,L2,pole):
    d=A-H; D=np.linalg.norm(d); Dm=(L1+L2)*0.9995
    short=0.0
    if D>Dm: short=D-Dm; D=Dm
    u=nrm(d); a=(L1*L1-L2*L2+D*D)/(2*D); h=math.sqrt(max(L1*L1-a*a,0))
    pp=np.asarray(pole,float); pp=pp-u*(pp@u); pp=nrm(pp)
    K=H+u*a+pp*h
    return K,H+u*D,pp,short
class Rig:
    def __init__(s,bones):
        # bones: name -> dict(head,tail,parent,Rrest_b(3x3 blender))
        s.b=bones
        g=lambda v: np.array([v[0],v[2],-v[1]])
        s.head={n:g(b['head']) for n,b in bones.items()}; s.tail={n:g(b['tail']) for n,b in bones.items()}
        s.d0={n:nrm(s.tail[n]-s.head[n]) for n in bones}
        s.L={n:np.linalg.norm(s.tail[n]-s.head[n]) for n in bones}
        s.Rrest_b={n:b['R'] for n,b in bones.items()}
        s.par={n:b['parent'] for n,b in bones.items()}
        up=np.array([0,1.,0]); fr=np.array([0,0,1.])
        s.s0={}
        for n in bones:
            if n.startswith('Foot'): s.s0[n]=up
            elif n.startswith('Hand'): s.s0[n]=np.array([-(1 if n.endswith('L') else -1),0,0.])
            else: s.s0[n]=fr
        s.names=list(bones)
    def solve(s,P):
        """P: dict of parameters -> (dict bone->Rdelta_g, hips_pos_g, info)"""
        D={}; pos={}; info={}
        D['Hips']=Eul(*P.get('hips',(0,0,0))); pos['Hips']=np.asarray(P['pelvis'],float)
        def child(n,par): pos[n]=pos[par]+D[par]@(s.head[n]-s.head[par])
        for n,par,key in (('Spine','Hips','spine'),('Chest','Spine','chest')):
            child(n,par); D[n]=Eul(*P.get(key,(0,0,0)))
        child('Neck','Chest'); D['Neck']=Eul(*P.get('neck',(0,0,0)))
        child('Head','Neck'); D['Head']=Eul(*P.get('head',(0,0,0)))
        for sd in 'LR':
            # legs
            ul,ll,ft='UpperLeg.'+sd,'LowerLeg.'+sd,'Foot.'+sd
            child(ul,'Hips'); H=pos[ul]
            LG=P['leg'+sd]
            A=np.asarray(LG['ankle'],float)
            K,A2,pp,short=ik2(H,A,s.L[ul],s.L[ll],LG['pole'])
            dU=nrm(K-H); dL=nrm(A2-K)
            D[ul]=aimR(s.d0[ul],s.s0[ul],dU,pp); D[ll]=aimR(s.d0[ll],s.s0[ll],dL,pp)
            pos[ll]=K; pos[ft]=A2
            D[ft]=Ry(math.radians(LG.get('yaw',0)))@Rx(math.radians(LG.get('pitch',0)))@Rz(math.radians(LG.get('roll',0)))
            info['leg'+sd]=dict(knee=K,ankle=A2,short=short,kneeflex=math.degrees(math.acos(np.clip(dU@dL,-1,1))))
            # arms
            ua,la,hd='UpperArm.'+sd,'LowerArm.'+sd,'Hand.'+sd
            child(ua,'Chest'); S=pos[ua]
            AR=P['arm'+sd]
            if AR.get('mode','world')=='chest': W=pos['Chest']+D['Chest']@np.asarray(AR['wrist'],float)
            else: W=np.asarray(AR['wrist'],float)
            E,W2,pp,short=ik2(S,W,s.L[ua],s.L[la],AR['pole'])
            dU=nrm(E-S); dL=nrm(W2-E)
            D[hd]=aimR(s.d0[hd],s.s0[hd],AR['hand'],AR['palm'])
            # forearm twist: half follows the hand (pronation), half follows the elbow pole
            f_up=-pp
            ch=D[hd]@s.s0[la]; ch=ch-dL*(ch@dL)
            f_lo=nrm(0.5*nrm(-pp-dL*(-pp@dL))+0.5*nrm(ch))
            D[ua]=aimR(s.d0[ua],s.s0[ua],dU,f_up); D[la]=aimR(s.d0[la],s.s0[la],dL,f_lo)
            pos[la]=E; pos[hd]=W2
            info['arm'+sd]=dict(elbow=E,wrist=W2,short=short)
        for n in ('Bust.L','Bust.R'): D[n]=D['Chest']
        for n in ('Glute.L','Glute.R'): D[n]=D['Hips']
        return D,pos,info
    def basis(s,D,pelvis):
        """convert world deltas to local pose-bone quaternions (blender frame) and hips location"""
        out={}
        Db={n:C@D[n]@C.T for n in D}
        for n in s.names:
            Rcr=s.Rrest_b[n]; p=s.par[n]; Dp=Db[p] if p else np.eye(3)
            Rb=Rcr.T@Dp.T@Db[n]@Rcr
            out[n]=Rb
        T=C@(np.asarray(pelvis)-s.head['Hips'])
        loc=s.Rrest_b['Hips'].T@T
        return out,loc
