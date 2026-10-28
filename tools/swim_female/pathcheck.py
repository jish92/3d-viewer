import bpy,sys,numpy as np,json
IN=sys.argv[sys.argv.index('--')+1]
bpy.ops.wm.open_mainfile(filepath=IN)
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
tracks=list(arm.animation_data.nla_tracks); arm.animation_data.action=None
res={}
for tr in tracks:
    for t in tracks: t.mute=(t.name!=tr.name)
    st=tr.strips[0]; f0,f1=int(st.frame_start),int(round(st.frame_end))
    P=[];Y=[]
    for f in range(f0,f1+1):
        bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
        M=arm.matrix_world@arm.pose.bones['Hips'].matrix
        P.append(list(M.translation)); 
        # forward = body +? : take hips local Y axis projected; also shoulders line
        ch=arm.matrix_world@arm.pose.bones['Chest'].matrix
        Y.append([M.col[0][0],M.col[0][1],ch.col[0][0],ch.col[0][1]])  # hips/chest local X (lateral) in world XY
    res[tr.name]=dict(p=P,x=Y)
json.dump(res,open('/tmp/path_'+sys.argv[-1]+'.json','w'))
