set -e
cd /workspace/me3d/swim_female
$BPY mk_base.py >/dev/null 2>&1
/workspace/me3d/venv/bin/python - <<'PY'
import json
from joints import J,y0
json.dump({k:(J[k]-[0,y0,0]).tolist() for k in json.load(open('v1_backup/joints_y.json'))},open('joints_y.json','w'))
PY
/workspace/me3d/venv/bin/python build_f.py full her_a.glb >/dev/null
/workspace/me3d/venv/bin/python build_parts.py | tail -1
/workspace/me3d/venv/bin/python meas.py stage_body.npz
