# Swimmer, one-piece (`swimher`) - build scripts

Original fictional adult woman swimmer, built from an MPFB2 / MakeHuman (CC0) base body in headless Blender.
Procedural vertex-coloured suit (navy, red/white trim, no logos), hair bun, eyes, brows, slides - no textures.
Scripts keep absolute paths under `/workspace/me3d/swim_female` (work dir of the original build) and need
`/workspace/me3d/bpyvenv` (bpy 5.x) and `/workspace/me3d/venv` (numpy/scipy/trimesh); they are kept here as documentation / for re-runs.

Pipeline: `mk_base.py` -> `build_f.py` (body + suit) -> `build_parts.py` (hair, bun, eyes, slides) -> `build_all.sh`
(simplify with `tools/simplify_swim.mjs`, `rig_f.py` 19-bone rig + Walk/Run, `retarget_mb_f.py` + `rig_f_mb.py` MotionBricks,
`bust_pass.py` bust spring bake, `export_f.py`) -> `node tools/optimize_anim.mjs` (DROP_MR=0.45) -> `models/swimmer-one-piece.glb`.

## Rig
17 bones as the jammer models + `Bust.L/R` (children of Chest) = 19 bones. Capsule-distance weights with graph smoothing;
the suit copies the weights of the skin underneath. `FRONT_HIP=0.9`: upper-front thigh skin/suit weight is moved toward the pelvis
so the thigh does not cut through the suit at high knee lift.

## MotionBricks has no gender / style conditioning
The SOMA/MotionBricks locomotion model is only conditioned on a heading/speed/style *mode* (idle, walk, jog ...) and its training
clips are a single neutral skeleton - there is no gender / body-type conditioning and no female reference clips. The female gait is
therefore applied during retargeting (`gait_params.json`, subtle on purpose):

| change | Walk | Run | MotionBricks |
|---|---|---|---|
| stride vs. original | x0.93 | x0.93 | x0.93 |
| step width (feet closer to midline) | ankles 0.4 in | 0.5 in | 7 mm narrower per foot |
| pelvis yaw (rotation) | 4.8 deg | 8.4 deg | x1.6 of the SOMA pelvis yaw |
| pelvic roll (lateral tilt/drop) | 2.8 deg | 2.8 deg | +1.6 deg |
| spine / chest counter-rotation | 0.85 / 0.95 of pelvis yaw | 0.85 / 1.0 | 0.15 / 0.30 yaw, 0.45 / 0.70 roll |
| arms | swing 21 deg, adducted 27 deg, elbow +3 | swing 40, adducted 41, elbow +20 | upper arm 4 deg / forearm 3 deg closer to the body |

## Bust secondary motion
Two helper bones weighted to the bust (smoothstep radius 35-115 mm, front only, torso only). Damped spring
(f = 4 Hz, zeta = 0.45) driven by the chest's vertical acceleration, baked into the keyframes. Peak: Walk 1.5 mm, MotionBricks 4.6 mm,
Run 7.0 mm. Max edge-length change in the bust after deformation 2.5 mm (Run), p99 edge strain 7.6 %.
