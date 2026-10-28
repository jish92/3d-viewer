import numpy as np,sys
from scipy.spatial import ConvexHull
def circ(V,y,tol=0.012,zr=None):
    m=np.abs(V[:,1]-y)<tol
    P=V[m][:,[0,2]]
    # torso only (exclude arms): |x|<0.25
    P=P[np.abs(P[:,0])<0.25] if y>0.9 else P
    h=ConvexHull(P); return h.area, np.ptp(P[:,0]), np.ptp(P[:,1])
def report(f):
    V=np.load(f)['Vb']
    # torso-only for hips/waist using verts with |x|<0.22
    out={}
    for n,y in (('bust',1.30),('underbust',1.22),('waist',1.08),('hips',0.92),('thigh',0.70)):
        P=V[np.abs(V[:,1]-y)<0.012]
        if n!='thigh': P=P[np.abs(P[:,0])<0.24]
        else: P=P[P[:,0]>0.0]
        P=P[:,[0,2]]; h=ConvexHull(P); out[n]=(round(h.area*100,1),round(np.ptp(P[:,0])*100,1),round(np.ptp(P[:,1])*100,1))
    # shoulder width: max |x| at y 1.38
    P=V[np.abs(V[:,1]-1.40)<0.015]; out['shoulder_w']=round(np.ptp(P[:,0])*100,1)
    out['height']=round(V[:,1].max(),3)
    print(f,out)
for f in sys.argv[1:]: report(f)
