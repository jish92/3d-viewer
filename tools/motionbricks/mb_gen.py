import sys,time,math,numpy as np; sys.path[:0]=['/workspace/me3d/mb','/workspace/me3d/mb/docs/examples']
import cpu_shim, warp as wp
import motionbricks.model.engine as me; me.GenerationTimer=cpu_shim.DummyTimer
import locomotion_loop_cpu as L
wp.init=lambda:None; wp.set_device=lambda d:None
mode=sys.argv[1]; out=sys.argv[2]; dur=float(sys.argv[3])
if mode=='default': pass
elif mode=='circle':
    # jog straight, ramp speed, then a full left circle back to original heading
    def steer(t):
        a=2*math.pi*min(max((t-1.5)/(dur-1.5-0.5),0),1)
        return np.array([math.cos(a),math.sin(a),0],dtype=np.float32)
    L.steering=steer
L.DURATION=dur
t0=time.time(); shown=L.run('soma'); print('gen time',time.time()-t0)
np.save(out,np.stack(shown))
