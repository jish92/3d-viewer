// Colour-aware simplification of the body mesh (geometry_1) before rigging. usage: node simplify_swim.mjs in.glb out.glb targetFaces
import { NodeIO } from '@gltf-transform/core';
import { weld } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
const [,, inp, out, tf] = process.argv;
await MeshoptSimplifier.ready;
const io = new NodeIO(); const doc = await io.read(inp);
await doc.transform(weld());
for (const mesh of doc.getRoot().listMeshes()) for (const prim of mesh.listPrimitives()) {
  const pos = prim.getAttribute('POSITION'); const col = prim.getAttribute('COLOR_0'); if (!col) continue;
  const P = pos.getArray(); const n = P.length/3; const I = Uint32Array.from(prim.getIndices().getArray());
  const colArr = col.getArray(); const cs = col.getElementSize(); const norm = col.getComponentType()===5121?255:(col.getComponentType()===5123?65535:1);
  const A = new Float32Array(n*3); for (let v=0; v<n; v++) for (let k=0;k<3;k++) A[v*3+k]=colArr[v*cs+k]/norm;
  const lock = new Uint8Array(n); for (let v=0; v<n; v++) if (P[v*3+1] > 1.5) lock[v]=1;
  // optional LOCK_BOX=xmin,xmax,ymin,ymax,zmin,zmax (can be repeated with ';'): keep full resolution there (e.g. the bust area that a helper bone deforms)
  if (process.env.LOCK_BOX) for (const box of process.env.LOCK_BOX.split(';')) { const [x0,x1,y0,y1,z0,z1]=box.split(',').map(Number); for (let v=0; v<n; v++) { const x=P[v*3],y=P[v*3+1],z=P[v*3+2]; if (x>=x0&&x<=x1&&y>=y0&&y<=y1&&z>=z0&&z<=z1) lock[v]=1; } }
  // optional LOCK_COLOR_EDGES=1: keep every vertex that sits on a colour boundary (trim stripes) so the trim edges stay clean
  if (process.env.LOCK_COLOR_EDGES && process.env.LOCK_COLOR_EDGES!=='0') { const T=+process.env.LOCK_COLOR_EDGES; let nl=0; const dif=(a,b)=>{ for (let k=0;k<3;k++) if (Math.abs(A[a*3+k]-A[b*3+k])>0.25) return true; return false; };
    for (let t=0; t<I.length; t+=3) for (let e=0;e<3;e++) { const a=I[t+e], b=I[t+(e+1)%3]; if (dif(a,b)) { if(!lock[a]){lock[a]=1;nl++;} if(!lock[b]){lock[b]=1;nl++;} } }
    console.log('colour-edge verts locked', nl); }
  const target = Math.min(Math.floor(+tf*3), I.length);
  const [res, err] = MeshoptSimplifier.simplifyWithAttributes(I, P, 3, A, 3, Array(3).fill(+(process.env.COLOR_W||2.0)), lock, target, 0.002, ['ErrorAbsolute']);
  console.log(mesh.getName(), 'faces', I.length/3, '->', res.length/3, 'err', err);
  prim.getIndices().setArray(res);
}
await doc.transform((d)=>{ // prune unreferenced vertices
  for (const mesh of d.getRoot().listMeshes()) for (const prim of mesh.listPrimitives()) {
    const I=prim.getIndices().getArray(); const used=new Map(); const order=[]; const NI=new Uint32Array(I.length);
    for (let i=0;i<I.length;i++){ let m=used.get(I[i]); if(m===undefined){m=order.length;used.set(I[i],m);order.push(I[i]);} NI[i]=m; }
    for (const sem of prim.listSemantics()) { const a=prim.getAttribute(sem); const es=a.getElementSize(); const src=a.getArray(); const dst=new src.constructor(order.length*es);
      for (let k=0;k<order.length;k++) for (let c=0;c<es;c++) dst[k*es+c]=src[order[k]*es+c]; a.setArray(dst); }
    prim.getIndices().setArray(NI);
  } });
await io.write(out, doc);
