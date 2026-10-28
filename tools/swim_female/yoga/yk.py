import numpy as np, math, copy
from ylib import *
import yclips, yposes as yp, kimo
from scipy.spatial.transform import Rotation as R, Slerp
FPS=30
def sm(u): u=min(max(u,0.0),1.0); return u*u*(3-2*u)
class KClip:
    def __init__(s,name,Ps,hold,breath=1.0): s.name=name; s.Ps=Ps; s.hold=hold; s.breath=breath; s.N=len(Ps); s.T=s.N/FPS
    def frames(s): return s.N
    def P(s,t):
        f=int(round(t*FPS)); P=copy.deepcopy(s.Ps[min(f,s.N-1)])
        return yclips.Clip.breathe(s,P,t)
def loopify(Ps,nb=12):
    """last nb frames cross-fade onto frame 0 so that frame N == frame 0"""
    N=len(Ps); out=[copy.deepcopy(p) for p in Ps]
    P0=Ps[0]
    for i in range(nb):
        f=N-nb+i; u=sm((i+1)/nb); out[f]=blendP(Ps[f],P0,u)
    return out
def blendP(A,B,u):
    out={}
    for k in A:
        a=A[k]; b=B[k]
        if isinstance(a,dict): out[k]=blendP(a,b,u)
        elif isinstance(a,str): out[k]=a
        elif isinstance(a,np.ndarray) and a.shape==(3,3):
            out[k]=Slerp([0,1],R.from_matrix([a,b]))([u]).as_matrix()[0]
        elif isinstance(a,np.ndarray): out[k]=a*(1-u)+b*u
        else: out[k]=a*(1-u)+b*u
    return out
def make(name,npz,rig,rest,blend_fn=None,cut=None,hold_frac=0.25,breath=1.0):
    Ps,info=kimo.retarget(npz,rig,rest,cut=cut)
    hy=np.array([p['_hips_y'] for p in Ps]); lo=hy.min(); hi=hy.max()
    deep=(hi-hy)/max(hi-lo,1e-6)
    if blend_fn: Ps=[blend_fn(P,float(deep[i])) for i,P in enumerate(Ps)]
    idx=np.where(deep>0.85)[0]; hold=(idx[0]/FPS,idx[-1]/FPS) if len(idx) else (0,0)
    Ps=loopify(Ps)
    return KClip(name,Ps,hold,breath),info
def squat_arms(rig,rest):
    S=yp.squat(rig,rest)
    def fn(P,deep):
        w=sm((deep-0.35)/0.45); out=copy.deepcopy(P)
        for sd in 'LR':
            a=P['arm'+sd]; b=S['arm'+sd]
            out['arm'+sd]=dict(mode='chest',wrist=a['wrist']*(1-w)+b['wrist']*w,pole=nrm(a['pole']*(1-w)+b['pole']*w),hand=nrm(a['hand']*(1-w)+b['hand']*w),palm=nrm(a['palm']*(1-w)+b['palm']*w))
        return out
    return fn
