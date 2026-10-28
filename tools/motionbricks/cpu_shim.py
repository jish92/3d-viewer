"""CPU shim for MotionBricks: replaces the TensorRT InferenceEngine with ONNX Runtime on CPU (warp kernels run on CPU device)."""
import sys, types, numpy as np
from unittest import mock
sys.modules.setdefault('tensorrt', mock.MagicMock())
import warp as wp
wp.config.quiet = True
wp.init(); wp.set_device('cpu')
import json, onnxruntime as ort
from pathlib import Path
import motionbricks.inference.engine as ie
from motionbricks.inference.types import TensorSpec, TensorMode

class OrtEngine:
    default_enable_tf32 = True
    def __init__(self, onnx_path, *, build_options=None, stream=None, cache_dir=None, force_rebuild=False, logger_severity='INFO', build_progress_monitor=None):
        self._closed=False; self._stream=None
        so=ort.SessionOptions(); so.intra_op_num_threads=8
        self.sess=ort.InferenceSession(str(onnx_path), so, providers=['CPUExecutionProvider'])
        npt={'tensor(float)':np.float32,'tensor(int32)':np.int32,'tensor(int64)':np.int64}
        specs=[]; self._buffers={}
        for mode,lst in ((TensorMode.INPUT,self.sess.get_inputs()),(TensorMode.OUTPUT,self.sess.get_outputs())):
            for t in lst:
                shp={'pred_global_motions':(1,64,366),'pred_num_tokens':(1,)}.get(t.name) or tuple(1 if not isinstance(d,int) else d for d in t.shape)
                dt=np.dtype(npt[t.type]); specs.append(TensorSpec(t.name,mode,shp,dt))
                self._buffers[t.name]=wp.zeros(shp,dtype=wp.dtype_from_numpy(dt),device='cpu')
        from types import MappingProxyType
        self._tensor_specs=MappingProxyType({s.name:s for s in specs})
        self._in=[t.name for t in self.sess.get_inputs()]; self._out=[t.name for t in self.sess.get_outputs()]
        self._cuda_graph=None
    device=property(lambda s: wp.get_device('cpu'))
    stream=property(lambda s: None)
    tensor_specs=property(lambda s:s._tensor_specs)
    cuda_graph_baked=property(lambda s:True)
    def device_tensor(self,name): return self._buffers[name]
    def enqueue_async(self):
        res=self.sess.run(self._out,{n:self._buffers[n].numpy() for n in self._in})
        for n,r in zip(self._out,res): self._buffers[n].assign(np.ascontiguousarray(r))
    def synchronize(self): pass
    def bake_cuda_graph(self): pass
    def _ensure_open(self): pass
    def close(self): self._closed=True
    def __enter__(self): return self
    def __exit__(self,*a): self.close()
ie.InferenceEngine=OrtEngine
import motionbricks.model.generation as gen
class DummyTimer:
    def __init__(self,stream=None): self.milliseconds={}
    from contextlib import contextmanager
    @contextmanager
    def stage(self,n): yield
    def read(self): pass
gen.GenerationTimer=DummyTimer
_orig_ss=wp.ScopedStream
wp.ScopedStream=lambda s: mock.MagicMock()
