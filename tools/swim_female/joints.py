from common import *
V,polys,G,E=load()
Y=yup(V)
J={}
for k,v in G.items():
    if k.startswith('joint-'):
        idx=[i for i,w in v]; J[k[6:]]=Y[idx].mean(0)
body=np.array(sorted(i for i,w in G['body'] if w>.5)); y0=Y[body][:,1].min()
if __name__=='__main__':
    print('y0',y0)
    for k in ['ground','pelvis','spine-1','spine-2','spine-3','spine-4','neck','head','head-2','l-clavicle','l-shoulder','l-elbow','l-hand','l-upper-leg','l-knee','l-ankle','l-foot-1','l-foot-2','l-eye','mouth','jaw','l-scapula']:
        print(k,(J[k]-[0,y0,0]).round(3))
