import bpy,sys,pickle,numpy as np
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index('--')+1])
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
r={}
for b in arm.data.bones:
    r[b.name]={'parent':b.parent.name if b.parent else None,'head':np.array(b.head_local),'tail':np.array(b.tail_local),'M':np.array(b.matrix_local)}
pickle.dump(r,open('/tmp/our_rest.pkl','wb')); print(len(r))
