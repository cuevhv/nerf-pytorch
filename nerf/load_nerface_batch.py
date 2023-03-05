import torch
from typing import List
import cv2
import imageio
import os, glob
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

def rescale_bbox(bbox, scale=1.):
    """bbox: [top, bottom, left, right]
    """
    center_height = (bbox[0]+bbox[1])/2
    center_width = (bbox[2]+bbox[3])/2
    bbox[:2] = (bbox[:2]-center_height) * scale  # top bottom
    bbox[2:] = (bbox[2:]-center_width) * scale  # left right
    bbox[:2] += center_height
    bbox[2:] += center_width
    bbox = np.clip(bbox, a_min=0, a_max=1)  # make sure it's within the image boundary 0-1
    return bbox


class NerfFaceDataset(Dataset):
    def __init__(self, basedir: str, load_expressions: bool = True, load_landmarks3d: bool = True, load_bbox: bool = True, split: str = 'train', bbox_scale: float= 2.0, preload: bool = False) -> None:
        self.basedir = basedir
        self.load_expressions = load_expressions
        self.load_landmarks3d = load_landmarks3d
        self.load_bbox = load_bbox
        self.bbox_scale = bbox_scale
        with open(os.path.join(basedir, f"transforms_{split}.json"), "r") as fp:
            metas = json.load(fp)
        self.frames = metas["frames"]
        self.intrinsics = metas["intrinsics"]
        self.camera_angle_x = metas["camera_angle_x"]


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.frames[idx]
        fname = os.path.join(self.basedir, frame["file_path"] + ".png")
        img_rgb = np.asarray(imageio.imread(fname)).astype(np.float32) / 255.0
        poses = np.array(frame["transform_matrix"])

        H, W = img_rgb.shape[:2]
        camera_angle_x = float(self.camera_angle_x)
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        intrinsics = np.array(self.intrinsics)
        hw = np.array([H, W])

        if self.load_expressions:
            expressions = np.array(frame["expression"])
        else:
            expressions = np.zeros(50)

        if self.load_bbox:
            bbox = np.array([frame["bbox"][1], frame["bbox"][3], frame["bbox"][0], frame["bbox"][2]])
            # We increase the scale of bbox as deca face detector bbox is around the face and not the head
            bbox = rescale_bbox(bbox, scale=self.bbox_scale)

        else:
            bbox = np.array([0.0, 1.0, 0.0, 1.0])

        bbox[0:2] *= H  # top, left
        bbox[2:4] *= W  # right, bottom
        bbox = np.floor(bbox)


        if self.load_landmarks3d:
            landmarks3d = np.array(frame["landmarks3d"])
        else:
            landmarks3d = np.array([0])


        sample = {"imgs": torch.from_numpy(img_rgb).float(),
                  "poses": torch.from_numpy(poses).float(),
                  "hw": torch.from_numpy(hw).float(),
                  "intrinsics": torch.from_numpy(intrinsics).float(),
                  "expressions": torch.from_numpy(expressions).float(),
                  "landmarks3d": torch.from_numpy(landmarks3d).float(),
                  "bbox": torch.from_numpy(bbox).int(),
                  "names": os.path.basename(fname)}

        return sample

if __name__ == "__main__":
    basedir='data/nerf_synthetic/face'
    dataset = NerfFaceDataset(basedir, load_expressions = True, load_landmarks3d = True, load_bbox = True, split = 'val', bbox_scale = 2.0)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    ldmks = []
    for data in tqdm(train_dataloader):
        ldmks.append(data["landmarks3d"])

    it = iter(train_dataloader)
    first = next(it)
    second = next(it)