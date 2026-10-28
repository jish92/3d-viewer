set -e
cd /workspace/me3d/swim_female
BPY=/workspace/me3d/bpyvenv/bin/python; PY=/workspace/me3d/venv/bin/python
(cd /workspace/3d-viewer && COLOR_W=12 LOCK_BOX="${LOCKB:--0.19,0.19,1.17,1.43,0.0,0.30}" node tools/simplify_swim.mjs /workspace/me3d/swim_female/her_full.glb /workspace/me3d/swim_female/out/her_s70000.glb 70000 | tail -1)
SIMP=out/her_s70000.glb $BPY rig_f.py -- out/her_rig.blend 2>&1 | grep -E "ground-plant|simp|SAVED|Error|Trace"
$BPY dump_rest.py -- out/her_rig.blend | tail -1
$PY retarget_mb_f.py /tmp/mb_circle.npy out/mbclip.npz loop 2>&1 | grep -E "pelvis|ankle|stride|loop"
$BPY rig_f_mb.py -- out/her_rig.blend out/her_mb.blend out/mbclip.npz 2>&1 | grep -E "ground|after|SAVED|Error|Trace"
$BPY bust_pass.py -- out/her_mb.blend out/her_final.blend 2>&1 | grep -E "raw|gains|peak|SAVED|Error|Trace"
$BPY export_f.py -- out/her_final.blend out/her_final.glb 2>&1 | grep -E "EXPORTED|Error|Trace"
