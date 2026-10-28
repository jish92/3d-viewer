import sys,numpy as np; sys.path[:0]=['/workspace/me3d/mb','/workspace/me3d/mb/docs/examples']
import cpu_shim, warp as wp
from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
sk=load_skeleton_from_usd(get_asset_file("human","soma_base_skel_minimal_no_root.usd"),SOMA_SKELETON_PRIM_PATH)
print([m for m in dir(sk) if not m.startswith('_')])
n=sk.num_joints; names=[sk.joint_name(i) for i in range(n)] if hasattr(sk,'joint_name') else None
print(n,names)
par=np.array(sk.parent_indices.numpy()); print(par)
ref=sk.reference_local_transforms.numpy(); print(ref.shape,ref[:3])
np.savez('/tmp/soma_skel.npz',names=np.array(names),par=par,ref=ref)
