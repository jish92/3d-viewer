set -e
cd /workspace/me3d/swim_female
/workspace/me3d/venv/bin/python build_f.py full her_a_t.glb | tail -1
/workspace/me3d/venv/bin/python build_parts.py | tail -1
export LOCKB="-0.19,0.19,1.17,1.50,-0.15,0.30;-0.19,0.19,0.84,1.0,-0.24,0.0"
export TARGET=${TARGET:-90000}
bash build_all.sh 2>&1 | grep -E "simp:|Error|Trace|EXPORTED"
cd /workspace/3d-viewer; DROP_MR=0.45 node tools/optimize_anim.mjs /workspace/me3d/swim_female/out/her_final.glb /workspace/me3d/swim_female/out/her_opt.glb 2>&1 | tail -1; ls -la /workspace/me3d/swim_female/out/her_opt.glb
cp /workspace/me3d/swim_female/out/her_opt.glb /workspace/3d-viewer/models/_tmp_her.glb
