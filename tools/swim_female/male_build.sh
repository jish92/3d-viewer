set -e
cd /workspace/me3d
venv/bin/python swim/build_swim.py brief swim/brief_full.glb full | tail -1
(cd /workspace/3d-viewer && node tools/simplify_swim.mjs /workspace/me3d/swim/brief_full.glb /workspace/me3d/swim/brief_s.glb 100000 | tail -2)
RIG_SRC=/workspace/me3d/swim/brief_s.glb bpyvenv/bin/python swim/rig_swim_run.py -- /workspace/me3d/swim/brief_run_rigged.glb 2>&1 | grep -E "EXPORTED|Error|Traceback"
bpyvenv/bin/python swim/rig_swim_mb.py -- /workspace/me3d/swim/brief_mb_rigged.glb /tmp/mb_rt.npz 2>&1 | grep -E "ground dz|after|EXPORTED|Error|Traceback"
(cd /workspace/3d-viewer && DROP_MR=0.45 node tools/optimize_anim.mjs /workspace/me3d/swim/brief_mb_rigged.glb /workspace/me3d/swim/brief_mb_opt.glb 2>&1 | tail -1)
ls -la swim/brief_mb_opt.glb
