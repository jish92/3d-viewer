import { NodeIO } from '@gltf-transform/core';
import { EXTMeshoptCompression, KHRMeshQuantization } from '@gltf-transform/extensions';
import { weld, simplify, normals, quantize, meshopt, dedup, prune } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptSimplifier, MeshoptDecoder } from 'meshoptimizer';
const [,, inp, out] = process.argv; // keepArg: comma-separated mesh names NOT to simplify
await MeshoptEncoder.ready; await MeshoptSimplifier.ready; await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions([EXTMeshoptCompression, KHRMeshQuantization]).registerDependencies({'meshopt.encoder': MeshoptEncoder,'meshopt.decoder': MeshoptDecoder});
const doc = await io.read(inp);
const smoothNormals = (doc) => {
  for (const mesh of doc.getRoot().listMeshes()) for (const prim of mesh.listPrimitives()) {
    const pos = prim.getAttribute('POSITION'), idx = prim.getIndices();
    const P = pos.getArray(), I = idx.getArray(), N = new Float32Array(P.length);
    // accumulate by position (not vertex index) so UV/colour seams don't create shading seams
    const key = i => Math.round(P[i]*1e5)+','+Math.round(P[i+1]*1e5)+','+Math.round(P[i+2]*1e5);
    const acc = new Map(), keys = new Array(P.length/3);
    for (let v=0; v<P.length/3; v++){ const k=key(v*3); keys[v]=k; if(!acc.has(k)) acc.set(k,[0,0,0]); }
    for (let t=0;t<I.length;t+=3){ const a=I[t]*3,b=I[t+1]*3,c=I[t+2]*3;
      const ux=P[b]-P[a],uy=P[b+1]-P[a+1],uz=P[b+2]-P[a+2],vx=P[c]-P[a],vy=P[c+1]-P[a+1],vz=P[c+2]-P[a+2];
      const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
      for (const k of [a,b,c]){ const e=acc.get(keys[k/3]); e[0]+=nx;e[1]+=ny;e[2]+=nz; } }
    for (let v=0; v<keys.length; v++){ const e=acc.get(keys[v]); const l=Math.hypot(e[0],e[1],e[2])||1; N[v*3]=e[0]/l;N[v*3+1]=e[1]/l;N[v*3+2]=e[2]/l; }
    prim.setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(N));
  }
};
const dropNormals = (doc) => { for (const mesh of doc.getRoot().listMeshes()) for (const prim of mesh.listPrimitives()) { const n = prim.getAttribute('NORMAL'); if (n) { prim.setAttribute('NORMAL', null); } } };
// Skinned/animated variant: weld + smooth normals + meshopt compression. No simplify and no quantize (KHR_mesh_quantization is not applied to skinned positions).
await doc.transform(dropNormals, weld(), smoothNormals, meshopt({encoder: MeshoptEncoder, level:'high'}));
for (const m of doc.getRoot().listMaterials()) { if (!m.getMetallicRoughnessTexture()) m.setRoughnessFactor(0.75).setMetallicFactor(0); }
await io.write(out, doc);
