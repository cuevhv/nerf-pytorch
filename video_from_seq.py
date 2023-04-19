"""python compare_outputs.py --gt_imgs_folder /home/hanz/Documents/projects/face_animation/datasets/nerface/person_2/test --pred_imgs_folder /home/hanz/Documents/projects/face_animation/nerf-pytorch/out/corrected_dataset/temp/face-small_ldmks3d_expression_cutoff_ldmks-enc-no-dist-no-dir_deform-code-body-bkground_corrected-dataset/fine --nerface_imgs_folder /home/hanz/Documents/projects/face_animation/nerf-pytorch/out/corrected_dataset/temp/nerface"""
import numpy as np
import ffmpeg
import cv2
from natsort import natsorted
import glob
import argparse
import os



def get_imgs_fns(dir_name):
    return natsorted(glob.glob(os.path.join(dir_name, "*.png")))


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--img_folder", type=str)
    args.add_argument("--out_name", type=str, default="out")
    return args.parse_args()


if __name__ == "__main__":
    args = parser()
    img_fns = get_imgs_fns(args.img_folder)
    img_bgr = cv2.imread(img_fns[0])

    save_vid = True
    if save_vid:
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(img_bgr.shape[1], img_bgr.shape[0]))
                .output(f"{args.out_name}.mp4", pix_fmt='yuv420p', tune='film', vcodec='libx264', r=30)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )

    for i, img_fn in enumerate(img_fns):
        img_bgr = cv2.imread(img_fn)#[:,:,::-1]
        if save_vid:
            process.stdin.write(
            img_bgr[:,:,::-1]
                .astype(np.uint8)
                .tobytes()
        )
        cv2.imshow("imgs", img_bgr)
        cv2.waitKey(1)


    process.stdin.close()
    process.wait()