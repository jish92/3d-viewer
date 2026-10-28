import numpy as np, scipy.sparse as sp
def smoothstep(x,a,b):
    t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def adjacency(F,n):
    E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(n,n)); A=(A+A.T).tocsr(); A.data[:]=1; return A
def vnormals(V,F):
    fn=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    N=np.zeros_like(V)
    for k in range(3): np.add.at(N,F[:,k],fn)
    return N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
def clip_mesh(V,F,f,attrs=None):
    """keep f>0 part; returns V2,F2,attrs2, f2 (f interpolated; new verts f=0), parent info"""
    attrs=[] if attrs is None else attrs
    f=np.asarray(f,float); ins=f>0
    k=ins[F].sum(1)
    newV=[];newA=[[] for _ in attrs]; newf=[]; emap={}
    nV=len(V)
    def edge(a,b):
        key=(a,b) if a<b else (b,a)
        if key in emap: return emap[key]
        t=f[a]/(f[a]-f[b])
        newV.append(V[a]+t*(V[b]-V[a]))
        for i,at in enumerate(attrs): newA[i].append(at[a]+t*(at[b]-at[a]))
        newf.append(0.0)
        emap[key]=nV+len(newV)-1; return emap[key]
    out=[]
    for tri in F[k==3]: out.append(tuple(tri))
    for tri in F[k==1]:
        i=[j for j in range(3) if ins[tri[j]]][0]; a=tri[i]; b=tri[(i+1)%3]; c=tri[(i+2)%3]
        out.append((a,edge(a,b),edge(a,c)))
    for tri in F[k==2]:
        o=[j for j in range(3) if not ins[tri[j]]][0]; c=tri[o]; a=tri[(o+1)%3]; b=tri[(o+2)%3]  # a,b inside; c outside
        ac=edge(a,c); bc=edge(b,c)
        out.append((a,b,bc)); out.append((a,bc,ac))
    V2=np.vstack([V]+[np.array(newV)]) if newV else V
    A2=[np.vstack([at,np.array(na)]) if na else at for at,na in zip(attrs,newA)]
    f2=np.concatenate([f,np.array(newf)])
    F2=np.array(out)
    # compact
    used=np.unique(F2); m=-np.ones(len(V2),int); m[used]=np.arange(len(used))
    return V2[used],m[F2],[a[used] for a in A2],f2[used],used
def icosphere(sub=3):
    t=(1+5**.5)/2
    V=np.array([(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),(0,-1,t),(0,1,t),(0,-1,-t),(0,1,-t),(t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)],float)
    F=np.array([(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),(3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)])
    V=V/np.linalg.norm(V,axis=1,keepdims=True)
    for _ in range(sub):
        cache={};Vl=list(V);Fn=[]
        def mid(a,b):
            k=(min(a,b),max(a,b))
            if k not in cache:
                m=(Vl[a]+Vl[b]); m/=np.linalg.norm(m); cache[k]=len(Vl); Vl.append(m)
            return cache[k]
        for a,b,c in F:
            ab,bc,ca=mid(a,b),mid(b,c),mid(c,a); Fn+= [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        V=np.array(Vl); F=np.array(Fn)
    return V,F
def subdivide(V,F,attrs=(),k=1):
    attrs=list(attrs)
    for _ in range(k):
        E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); E.sort(1); E,inv=np.unique(E,axis=0,return_inverse=True)
        n=len(V); nf=len(F)
        mid=(V[E[:,0]]+V[E[:,1]])/2
        attrs=[np.vstack([a,(a[E[:,0]]+a[E[:,1]])/2]) for a in attrs]
        e01=inv[:nf]+n; e12=inv[nf:2*nf]+n; e20=inv[2*nf:]+n
        a,b,c=F[:,0],F[:,1],F[:,2]
        F=np.concatenate([np.c_[a,e01,e20],np.c_[b,e12,e01],np.c_[c,e20,e12],np.c_[e01,e12,e20]])
        V=np.vstack([V,mid])
    return V,F,attrs
