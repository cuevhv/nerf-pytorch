from utils.face_parsing.model import BiSeNet as model

import torch

import os
import os.path as osp
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from typing import List

class BiseNet(object):
    def __init__(self, is_bgr, weight_pth, device='cuda'):
        self.is_bgr = is_bgr
        self.device = device

        self.idx2name= {0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 9: 'ear_r',
                    10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck', 15: 'neck_l', 16: 'cloth', 17: 'hair', 18: 'hat'}
        self.name2idx = {name: idx for idx, name in self.idx2name.items()}

        n_classes = 19
        self.net = model(n_classes=n_classes).to(device)
        self.net.load_state_dict(torch.load(weight_pth))
        self.net.eval()


    def mask_img(self, img, bg_list: List[str]):
        for idx in bg_list:
            img[img == idx] = 0
        img[img > 0] = 1
        return img



    def preprocess_img(self, img):
        to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        if type(img) == torch.Tensor:
            img = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))
        if self.is_bgr:
            img = img[:,:,::-1]  # bgr -> rgb


        h, w = img.shape[:2]

        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        return img, (h, w)


    def infer(self, img):
        with torch.no_grad():
            img, (h, w) = self.preprocess_img(img)
            img = img.to(self.device)
            out = self.net(img)[0]
            parsing = cv2.resize(out.squeeze(0).cpu().numpy().argmax(0), (w,h), interpolation=cv2.INTER_NEAREST)

            # plt.subplot(1,2,1), plt.imshow(parsing)
            parsing = self.mask_img(parsing, ['cloth', 'hat'])
            # plt.subplot(1,2,2), plt.imshow(parsing)
            # plt.show()

        return parsing




def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



