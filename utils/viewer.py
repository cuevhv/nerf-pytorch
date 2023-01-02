import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import copy

def show_dirs(poses, cfg):
    """
    poses w2c in homogenous coordinates Nx4x4 matrix [R|t]
    cfg: cfg variables
    """
    print(poses.shape)
    dirs = -poses[:, :3, 2]  # select vector -z from pose matrix [x,y,z|t] 
    origins = poses[:, :3, 3]

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
    origins[..., 0].flatten(),
    origins[..., 1].flatten(),
    origins[..., 2].flatten(),
    dirs[..., 0].flatten(),
    dirs[..., 1].flatten(),
    dirs[..., 2].flatten(), length=0.5, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()

    show_camera_coords(poses)

def show_camera_coords(poses):
    """ Shows the poses of the camera coordinates, useful to know what is the camera convension
        poses w2c in homogenous coordinates Nx4x4 matrix [R|t]
        cfg: cfg variables
    """
    coord_mesh_orgin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coord_meshes_cams = []
    n = poses.shape[0]

    for i in range(n):
        coord_mesh_cam = copy.deepcopy(coord_mesh_orgin).transform(poses[i])
        coord_meshes_cams.append(coord_mesh_cam)
    o3d.visualization.draw_geometries(coord_meshes_cams)
