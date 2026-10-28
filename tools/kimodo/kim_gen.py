import os,sys,torch,json,numpy as np
sys.path.insert(0,'/workspace/me3d/kimodo')
from kimodo import load_model
from kimodo.exports.motion_io import save_kimodo_npz
names=sys.argv[1:]
P=json.load(open(os.environ.get('KP','/workspace/me3d/kim_prompts.json'))); E=torch.load('/workspace/me3d/kim_emb.pt')
import re
N=lambda t:re.sub(r'[^a-z0-9]','',t.lower())
EN={N(k):v for k,v in E.items()}
class Enc:
    def __call__(self,texts):
        fs=[EN[N(t)][0] for t in texts]; return torch.stack(fs) if fs[0].dim()==2 else torch.stack(fs), [EN[N(t)][1] for t in texts]
    def to(self,d): return self
    def eval(self): return self
model=load_model('Kimodo-SOMA-RP-v1.2',device='cpu',text_encoder=Enc())
print('fps',model.fps)
for n in names:
    p=P[n]; torch.manual_seed(p.get('seed',0))
    out=model(p['texts'],p['frames'],constraint_lst=[],num_denoising_steps=100,num_samples=1,multi_prompt=True,num_transition_frames=5,post_processing=False,return_numpy=True)
    single={k:(v[0] if hasattr(v,'shape') and len(v.shape)>0 and v.shape[0]==1 else v) for k,v in out.items()}
    os.makedirs('/workspace/me3d/kim_out',exist_ok=True); save_kimodo_npz('/workspace/me3d/kim_out/%s.npz'%n,single); print('GEN',n,{k:getattr(v,'shape',None) for k,v in single.items()},flush=True)
