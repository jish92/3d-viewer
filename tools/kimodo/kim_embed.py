import os,sys,torch,json
os.environ['TEXT_ENCODERS_DIR']='/workspace/me3d/kimodo_models'
os.environ['TEXT_ENCODER_DEVICE']='cpu'
sys.path.insert(0,'/workspace/me3d/kimodo')
from kimodo.model import LLM2VecEncoder
P=json.load(open('/workspace/me3d/kim_prompts.json'))
texts=sorted({t for v in P.values() for t in v['texts']})
enc=LLM2VecEncoder('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised','bfloat16',4096,'cpu')
out={}
for t in texts:
    e,l=enc(t); out[t]=(e.float().cpu(),l); print('enc',t[:50],tuple(e.shape),flush=True)
torch.save(out,'/workspace/me3d/kim_emb.pt'); print('EMB_DONE')
