# 3d-viewer

Tiny static 3D model viewer (Google `<model-viewer>`, vendored — no build step, no CDN).

- `?m=<id>` opens a model from `models.json`; `?model=<url>` opens any GLB/GLTF/OBJ URL (server must send CORS headers).
- Drag & drop / file picker for local `.glb`, `.gltf` (self-contained) and `.obj` (converted to GLB in the browser with three.js).

## Add a model
1. (Optional) shrink it: `npm i && node tools/optimize.mjs input.glb models/name.glb 0.35`  
   (weld → simplify to 35% of faces → smooth normals → quantize → meshopt; keeps vertex colors. Use ratio `1` for no simplification.)
2. Put the file in `models/`, add an entry to `models.json` (`id`, `label`, `url`, optional `note`, optional `full` = path to full-res version).
3. Commit & push; GitHub Pages redeploys in ~1 min.
