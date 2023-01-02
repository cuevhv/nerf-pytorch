import json
import os

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_nerface_data(basedir, half_res=False, testskip=1, debug=False,
                      load_expressions=True, load_bbox=True):
    """ based on 
    https://github.com/gafniguy/4D-Facial-Avatars/blob/977606261b8d7e551dd455d66cd187d0d23c5a75/nerface_code/nerf-pytorch/nerf/load_flame.py
    """
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_expressions = []
    all_landmarks3d = []
    all_bboxs = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        expressions = []
        bboxs = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        print(f"loading {s}")
        for i, frame in tqdm(enumerate(meta["frames"][::skip])):
            # if i > 100: break
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))

            if load_expressions:
                expressions.append(np.array(frame["expression"]))
            else:
                expressions.append(np.zeros(50)) # we have 50 expressions from DECA
            
            # bbox deca [left, top, right, bottom] -> [top, bottom, left, right] 
            if load_bbox:
                bboxs.append(np.array([frame["bbox"][1], frame["bbox"][3], frame["bbox"][0], frame["bbox"][2]]))
            else:
                bboxs.append(np.array([0.0, 1.0, 0.0, 1.0])) 

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)

        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if meta["intrinsics"]:
        intrinsics = np.array(meta["intrinsics"])
    else:
        intrinsics = np.array[[focal, focal, 0.5, 0.5]]  # fx fy cx cy
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    # In debug mode, return extremely tiny images
    if debug:
        H = H // 32
        W = W // 32
        focal = focal / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        poses = torch.from_numpy(poses)
        return imgs, poses, render_poses, [H, W, focal], i_split

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 2
        W = W // 2
        # focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * 0.5
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
    
    else:
        imgs = [
            torch.from_numpy(imgs[i]
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    bboxs[:,0:2] *= H  # top, left
    bboxs[:,2:4] *= W  # right, bottom
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()

    print("finish loading")

    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions
