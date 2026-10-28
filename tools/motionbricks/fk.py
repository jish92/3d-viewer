import numpy as np
def qmul(a,b):
    ax,ay,az,aw=a[...,0],a[...,1],a[...,2],a[...,3]; bx,by,bz,bw=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz],-1)
def qrot(q,v):
    qv=q[...,:3]; w=q[...,3:4]; t=2*np.cross(qv,v); return v+w*t+np.cross(qv,t)
def fk(local,par):
    # local [...,J,7] pos+quat(xyzw)
    J=local.shape[-2]; P=np.zeros(local.shape[:-1]+(3,)); Q=np.zeros(local.shape[:-1]+(4,))
    for j in range(J):
        p=par[j]
        if p<0: P[...,j,:]=local[...,j,:3]; Q[...,j,:]=local[...,j,3:]
        else:
            P[...,j,:]=P[...,p,:]+qrot(Q[...,p,:],local[...,j,:3]); Q[...,j,:]=qmul(Q[...,p,:],local[...,j,3:])
    return P,Q
