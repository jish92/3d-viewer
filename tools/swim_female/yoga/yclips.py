# clip timelines for the yoga poses: keys -> per-frame parameter sets (python only, no bpy)
import numpy as np, math, copy
from ylib import *
import yposes as yp
FPS=30
def ss(u): u=min(max(u,0.0),1.0); return u*u*u*(u*(u*6-15)+10)      # smootherstep
def lerp(a,b,u):
    if isinstance(a,dict):
        out={}
        for k in set(a)|set(b):
            if k in a and k in b: out[k]=lerp(a[k],b[k],u)
            else:
                v=a[k] if k in a else b[k]
                out[k]=lerp(v,(0,0,0),u) if (k in a and isinstance(v,tuple)) else (lerp((0,0,0),v,u) if isinstance(v,tuple) else v)
        return out
    if isinstance(a,str): return a if u<0.5 else b
    if isinstance(a,(tuple,list)): return tuple(np.asarray(a,float)*(1-u)+np.asarray(b,float)*u)
    return a*(1-u)+b*u
def mkP(Pa,Pb,u,arcs=None):
    P=lerp(Pa,Pb,u)
    if arcs:
        for k,h in arcs.items():
            P[k]=dict(P[k]); key='ankle' if k.startswith('leg') else 'wrist'
            a=np.array(P[k][key],float); a[1]+=h*math.sin(math.pi*u); P[k][key]=a
    return P
class Clip:
    def __init__(s,name,keys,hold,breath=1.0,loop_wrap=True):
        """keys: list of (t, P, arcs-for-segment-ending-here); hold=(t0,t1) the pose hold window used for breathing weight"""
        s.name=name; s.keys=keys; s.hold=hold; s.breath=breath; s.T=keys[-1][0]
    def P(s,t):
        K=s.keys
        for i in range(len(K)-1):
            if K[i][0]<=t<=K[i+1][0]:
                u=(t-K[i][0])/max(K[i+1][0]-K[i][0],1e-9)
                P=mkP(K[i][1],K[i+1][1],ss(u),K[i+1][2])
                return s.breathe(P,t)
        return s.breathe(copy.deepcopy(K[-1][1]),t)
    def breathe(s,t,tt=None):
        P=t; t=tt
        w=0.35+0.65*min(ss((t-s.hold[0]+0.8)/0.8),ss((s.hold[1]+0.8-t)/0.8))     # stronger breathing in the hold
        ph=2*math.pi*t/4.2
        b=math.sin(ph)*w*s.breath
        for k,g in (('chest',1.6),('spine',0.7),('neck',-0.6),('head',-0.4)):
            e=np.array(P.get(k,(0,0,0)),float); e[1]+=g*b; P[k]=tuple(e)
        return P
    def frames(s): return int(round(s.T*FPS))        # loop: last frame != first frame (frame N == frame 0)
def seq(keys,mirror=True):
    """forward keys [(dt,P,arcs)] -> there-and-back list with absolute times"""
    pass
def there_and_back(fwd,hold_t,t0=0.0):
    """fwd: list of (dur, P, arcs) ; first entry is the start pose (dur ignored)."""
    keys=[]; t=t0
    keys.append((t,fwd[0][1],None))
    for dur,P,arcs in fwd[1:]:
        t+=dur; keys.append((t,P,arcs))
    th0=t; t+=hold_t; keys.append((t,fwd[-1][1],None)); th1=t
    for i in range(len(fwd)-1,0,-1):
        t+=fwd[i][0]; keys.append((t,fwd[i-1][1],fwd[i][2]))
    return keys,(th0,th1)
