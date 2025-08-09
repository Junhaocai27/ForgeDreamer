import aspose.threed as a3d

scene = a3d.Scene.from_file("/home/s414e2/CJH/Text-to-3D/LucidDreamer/output/bolt3/init_points3d.ply")
scene.save("Output.obj")
