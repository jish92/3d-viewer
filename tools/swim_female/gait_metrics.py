import bpy,numpy as np,sys,json
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index('--')+1])
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]; pb=arm.pose.bones
tracks={t.name:t for t in arm.animation_data.nla_tracks}
def H(n): return np.array(pb[n].head); 
def T(n): return np.array(pb[n].tail)
for name in [n for n in ('Walk','Run','MotionBricks') if n in tracks]:
    for n,t in tracks.items(): t.mute=(n!=name)
    st=tracks[name].strips[0]; fr=range(int(st.frame_start),int(st.frame_end)+1)
    ua=[];la=[];sw=[];pelv=[];ch=[];roll=[];fwd=[]
    for f in fr:
        bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
        for sd,s in (('L',1),('R',-1)):
            v=T('UpperArm.'+sd)-H('UpperArm.'+sd); ua.append(np.degrees(np.arctan2(abs(v[0]),-v[2])))   # abduction from vertical, frontal plane
            v=T('LowerArm.'+sd)-H('LowerArm.'+sd); la.append(np.degrees(np.arctan2(abs(v[0]),-v[2])))
        sw.append(abs(H('Foot.L')[0]-H('Foot.R')[0])); fwd.append(abs(H('Foot.L')[1]-H('Foot.R')[1]))
        h=H('UpperLeg.L')-H('UpperLeg.R'); pelv.append(np.degrees(np.arctan2(h[1],h[0]))); roll.append(np.degrees(np.arctan2(h[2],h[0])))
        s_=H('UpperArm.L')-H('UpperArm.R'); ch.append(np.degrees(np.arctan2(s_[1],s_[0])))
    pelv=np.array(pelv);ch=np.array(ch);roll=np.array(roll)
    print(f'GAIT {name}: ankle-ankle lateral mean {np.mean(sw)*100:.1f} cm, fore-aft separation max {np.max(fwd)*100:.1f} cm, pelvis yaw pk-pk {np.ptp(pelv-np.median(pelv)):.1f} deg, pelvic tilt pk-pk {np.ptp(roll):.1f} deg, shoulder-line yaw pk-pk {np.ptp(ch):.1f} deg, upper-arm abduction mean {np.mean(ua):.1f} deg, forearm abduction mean {np.mean(la):.1f} deg')
