# female_rigged.glb – rigged female character (bikini), neutral standing pose

Standalone, reusable asset (no chair, no second character, no animation clips).

## Format / units
- glTF 2.0 binary (`female_rigged.glb`), ~2.9 MB, no textures (uncompressed, no Draco/meshopt extensions).
- **Units: metres** (character is ~1.79 m tall including the hair bun; feet at y = 0).
- **Up axis: +Y** (glTF standard). Character **faces +Z**; her left side is at **+X**.
- Rest pose = neutral standing A-pose (arms ~40° from the body, feet flat on the floor).
- Origin is on the floor between the feet.

## Meshes (2 skinned meshes, 1 skin / 1 armature)
| Mesh | Tris | Contents |
|---|---|---|
| `Body` | ~58.8k (30.6k verts) | skin, face (eyes, brows, lips, smile), hair bun, sandals, hidden hip-seam filler patch |
| `Bikini` | ~23.3k (13.8k verts) | red triangle top, side-tie bottoms, neck/back ties, cords + tassels |

- Colours are **vertex colours** (`COLOR_0`), no texture maps. Materials are plain Principled (metallic 0; roughness 0.5 body / 0.38 bikini).
- Bikini is a separate mesh so it can be hidden/swapped; it shares the same skeleton and weights field.
- Max 4 bone influences per vertex (standard linear-blend skinning).

## Skeleton (22 bones, Mixamo-style names without the `mixamorig:` prefix)
```
Hips
├─ Spine ─ Spine1 ─ Spine2
│            ├─ Neck ─ Head
│            ├─ LeftShoulder ─ LeftArm ─ LeftForeArm ─ LeftHand
│            └─ RightShoulder ─ RightArm ─ RightForeArm ─ RightHand
├─ LeftUpLeg ─ LeftLeg ─ LeftFoot ─ LeftToeBase
└─ RightUpLeg ─ RightLeg ─ RightFoot ─ RightToeBase
```
- Left bones are on +X. Bones are a pure hierarchy of joint-to-joint bones (no IK/twist/helper bones, no end bones; fingers are not rigged – hands are rigid on `LeftHand`/`RightHand`).
- Hip joint height is 0.885 m (Hips bone head is the pelvis centre at the same height).
- Weights: thighs are ~100 % `UpLeg` between hip and knee with narrow (≈±4.5 cm hip / ±3.2 cm knee / ±3 cm ankle) blend bands, position-based analytic weights in the leg system; torso/arms use label + geodesic narrow bands. Garment weights come from the same field so the bikini follows the skin.
- Tested through a sit/walk animation with linear blend skinning (no dual-quaternion needed).

## Shape keys (morph targets)
- `Body` has one morph target: **`Smile`** (0..1, default 0). 0 = relaxed neutral mouth, 1 = full friendly smile (mouth corners up/back, cheeks lifted). Only the face vertices move. (In the `couple-rest` viewer scene the smile is baked in instead.)
- `Bikini` has no morph targets.

## Known limitations
- Hand/fingers and toes are not individually rigged (ToeBase bone exists for the sandal sole/foot flex).
- Hair is a rigid bun mesh weighted to `Head`.
- Eyebrow geometry is a little crude; skin has a faint hairline seam at the hip where the original parts meet (hidden by the bikini).
- Flat vertex-colour shading; add your own textures/materials if needed.
