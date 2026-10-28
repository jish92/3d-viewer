import bpy,sys
SRC,OUT=sys.argv[sys.argv.index('--')+1:sys.argv.index('--')+3]
bpy.ops.wm.open_mainfile(filepath=SRC)
for o in bpy.data.objects: o.select_set(True)
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',use_selection=True,export_animations=True,export_skins=True,
    export_force_sampling=True,export_frame_range=False,export_animation_mode='ACTIONS',export_yup=True,export_vertex_color='ACTIVE',
    export_apply=False,export_image_format='AUTO',export_optimize_animation_size=False)
print('EXPORTED',OUT)
