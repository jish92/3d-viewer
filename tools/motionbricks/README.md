CPU-only MotionBricks (NVlabs/motionbricks, SOMA ONNX) -> swim-brief retarget pipeline.
1. cpu_shim.py: replaces the TensorRT engine with ONNX Runtime on CPU (warp kernels run on CPU).
2. mb_gen.py circle out.npy 9: runs docs/examples/locomotion_loop.py (steering = jog, then a full circle).
3. sk.py dumps the SOMA skeleton, fk.py is the FK helper, retarget_mb.py maps SOMA joints to the 17-bone rig (loop-matched, crossfaded).
4. rig_swim_mb.py writes the 'MotionBricks' action into the rigged blend and exports the GLB.
