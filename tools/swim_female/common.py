import numpy as np, json
def load():
    d=np.load('/workspace/me3d/swim_female/full_raw.npz',allow_pickle=True)
    V=d['V']; polys=list(d['polys']); G=json.load(open('/workspace/me3d/swim_female/groups.json'))
    E=np.load('/workspace/me3d/swim_female/extra_raw.npz',allow_pickle=True)
    return V,polys,G,E
def yup(P): return np.stack([P[:,0],P[:,2],-P[:,1]],-1)
def tri(polys):
    t=[]
    for p in polys:
        for i in range(1,len(p)-1): t.append((p[0],p[i],p[i+1]))
    return np.array(t)
