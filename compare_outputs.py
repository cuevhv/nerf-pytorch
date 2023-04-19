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
    args.add_argument("--gt_imgs_folder", type=str)
    args.add_argument("--pred_imgs_folder", type=str)
    args.add_argument("--nerface_imgs_folder", type=str)
    return args.parse_args()


if __name__ == "__main__":
    args = parser()
    gt_fns = get_imgs_fns(args.gt_imgs_folder)
    pred_fns = get_imgs_fns(args.pred_imgs_folder)
    nerface_fns = get_imgs_fns(args.nerface_imgs_folder)

    save_vid = True
    if save_vid:
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(512*3, 512))
                .output("out_vid_mine_vs_mine_iris_mask.mp4", pix_fmt='yuv420p', tune='film', vcodec='libx264', r=30)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )

    from utils.face_parsing.bisenet import BiseNet
    import requests
    weight_path = "utils/face_parsing/79999_iter.pth"
    if not os.path.exists(weight_path):
        print("downloading weights for face parsing")
        url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download"
        response = requests.get(url)
        open(weight_path, "wb").write(response.content)
    face_seg_net = BiseNet(True, weight_path)


    old_img = 0

    for i, (gt_fn, pred_fn, nerface_fn) in enumerate(zip(*(gt_fns, pred_fns, nerface_fns))):
        img_gt = cv2.imread(gt_fn)#[:,:,::-1]
        out_gt = face_seg_net.infer(img_gt.astype(np.float32)/255.).astype(np.float32)
        img_gt = (img_gt*out_gt[:,:,None]+(1-out_gt[:,:, None])).astype(np.uint8)

        img_pred = cv2.imread(pred_fn)
        out = face_seg_net.infer(img_pred.astype(np.float32)/255.).astype(np.float32)
        img_pred = (img_pred*out_gt[:,:,None]*out[:,:,None]+(1-out_gt[:,:,None]*out[:,:, None])).astype(np.uint8)

        img_nerface = cv2.imread(nerface_fn)
        # if i > 0:
        #     img_nerface = (img_nerface*0.6 + old_img*0.4).astype(np.uint8)
        # old_img = img_nerface
        out = face_seg_net.infer(img_nerface.astype(np.float32)/255.).astype(np.float32)
        img_nerface = (img_nerface*out_gt[:,:,None]*out[:,:,None]+(1-out_gt[:,:,None]*out[:,:, None])).astype(np.uint8)

        cv2.putText(img=img_gt, text='GT', org=(10, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 255, 0),thickness=1)
        cv2.putText(img=img_pred, text='iris', org=(10, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 255, 0),thickness=1)
        cv2.putText(img=img_nerface, text='nerface', org=(10, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 255, 0),thickness=1)

        img = np.hstack((img_gt, img_nerface, img_pred))
        if save_vid:
            process.stdin.write(
            img[:,:,::-1]
                .astype(np.uint8)
                .tobytes()
        )
        cv2.imshow("imgs", img)
        cv2.waitKey(1)

    # video.release()



    process.stdin.close()
    process.wait()