import { NodeIO } from '@gltf-transform/core';
import { EXTMeshoptCompression, KHRMeshQuantization } from '@gltf-transform/extensions';
import { weld, simplify, normals, quantize, meshopt, dedup, prune } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptSimplifier, MeshoptDecoder } from 'meshoptimizer';
const [,, inp, out, ratio] = process.argv;
await MeshoptEncoder.ready; await MeshoptSimplifier.ready; await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions([EXTMeshoptCompression, KHRMeshQuantization]).registerDependencies({'meshopt.encoder': MeshoptEncoder,'meshopt.decoder': MeshoptDecoder});
const doc = await io.read(inp);
const smoothNormals = (doc) => {
  for (const mesh of doc.getRoot().listMeshes()) for (const prim of mesh.listPrimitives()) {
    const pos = prim.getAttribute('POSITION'), idx = prim.getIndices();
    const P = pos.getArray(), I = idx.getArray(), N = new Float32Array(P.length);
    for (let t=0;t<I.length;t+=3){ const a=I[t]*3,b=I[t+1]*3,c=I[t+2]*3;
      const ux=P[b]-P[a],uy=P[b+1]-P[a+1],uz=P[b+2]-P[a+2],vx=P[c]-P[a],vy=P[c+1]-P[a+1],vz=P[c+2]-P[a+2];
      const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
      for (const k of [a,b,c]){N[k]+=nx;N[k+1]+=ny;N[k+2]+=nz;} }
    for (let i=0;i<N.length;i+=3){const l=Math.hypot(N[i],N[i+1],N[i+2])||1;N[i]/=l;N[i+1]/=l;N[i+2]/=l;}
    prim.setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(N));
  }
};
const steps = [weld()];
if (+ratio < 1) steps.push(simplify({ simplifier: MeshoptSimplifier, ratio: +ratio, error: 0.01, lockBorder: false }));
steps.push(smoothNormals, quantize({quantizePosition:14, quantizeNormal:10, quantizeColor:8}), meshopt({encoder: MeshoptEncoder, level:'high'}));
await doc.transform(...steps);
// nicer default material
for (const m of doc.getRoot().listMaterials()) { m.setRoughnessFactor(0.75).setMetallicFactor(0); }
await io.write(out, doc);
