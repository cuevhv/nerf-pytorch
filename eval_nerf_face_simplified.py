import argparse
import os
import time

import imageio
import numpy as np
import torch
from PIL import Image
import torchvision
import yaml
from tqdm import tqdm

from nerf import (
    CfgNode,
    load_nerface_data,
    get_ray_bundle_nerface,
    RefinePose,
    get_ray_bundle,
    load_blender_data,
    load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf, NerfFaceDataset,
    NerfBase,
)


def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--savedir", type=str, help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "face":
        images, poses, render_poses, hwf, i_split, expressions, landmarks3d, bboxs, names = load_nerface_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            load_expressions=cfg.dataset.use_expression,
            load_landmarks3d=cfg.dataset.use_landmarks3d,
        )
        # show_dirs(poses, cfg)

        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

    elif cfg.dataset.type.lower() == "face_dataloader":
        train_dataset = NerfFaceDataset(cfg.dataset.basedir, load_expressions = cfg.dataset.use_expression,
                                                        load_landmarks3d = cfg.dataset.use_landmarks3d,
                                    load_bbox = True, split = 'train', bbox_scale = 2.0)

        test_dataset = NerfFaceDataset(cfg.dataset.basedir, load_expressions = cfg.dataset.use_expression,
                                                        load_landmarks3d = cfg.dataset.use_landmarks3d,
                                    load_bbox = True, split = 'test', bbox_scale = 2.0)
        # eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
        i_train, i_test = (list(range(len(train_dataset))), list(range(len(test_dataset))))
        use_dataloader = True
        H, W = test_dataset[0]["hw"].numpy()
        print(H,W)
        H, W = int(H), int(W)
        focal = test_dataset[0]["intrinsics"].numpy()
        print(focal)
    else:
        raise NotImplementedError

    if not cfg.dataset.use_expression:
        expressions = None

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    if hasattr(cfg.dataset, "mask_face") and cfg.dataset.mask_face:
        from utils.face_parsing.bisenet import BiseNet
        import requests
        weight_path = "utils/face_parsing/79999_iter.pth"
        if not os.path.exists(weight_path):
            print("downloading weights for face parsing")
            url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download"
            response = requests.get(url)
            open(weight_path, "wb").write(response.content)
        face_seg_net = BiseNet(False, weight_path)


    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_ldmks_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_ldmks,
        include_input=cfg.models.coarse.include_input_ldmks,
        log_sampling=cfg.models.coarse.log_sampling_ldmks,
    )

    encode_ldmks_dir_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir_ldmks,
        include_input=cfg.models.coarse.include_input_ldmks,
        log_sampling=cfg.models.coarse.log_sampling_ldmks,
        encoding_type=cfg.nerf.encode_ldmks_direction_fn \
                    if hasattr(cfg.nerf, "encode_ldmks_direction_fn") else "none",
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    if cfg.dataset.fix_background:
        # based on nerface https://github.com/gafniguy/4D-Facial-Avatars/blob/989be64216df754a4a34f8f53d7a71af130b57d5/nerface_code/nerf-pytorch/train_transformed_rays.py#L160
        print("loading gt background to condition on")
        background_img = Image.open(os.path.join(cfg.dataset.basedir,'bg','00050.png'))
        background_img.thumbnail((H,W))
        background_img = torch.from_numpy(np.array(background_img).astype(np.float32)).to(device)
        background_img = background_img/255
        print("bg shape", background_img.shape)
        # print("should be ", images[i_train][0].shape)
        # assert background_img.shape == images[i_train][0].shape
    elif hasattr(cfg.dataset, "mask_face") and cfg.dataset.mask_face:
        background_img = torch.ones((H,W,3)).float().to(device)
        # assert background_img.shape == images[i_train][0].shape
    else:
        background_img = None

    # Initialize nerf network
    nerf_network = NerfBase(len(i_train), device)
    trainable_parameters = nerf_network.init_network(cfg, models)

    # Initialize optimizer.
    use_amp = hasattr(cfg.optimizer, "use_amp") and cfg.optimizer.use_amp
    print("using amp:", use_amp)

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.checkpoint):
        checkpoint = torch.load(configargs.checkpoint)
        nerf_network.load_checkpoint(checkpoint, None)
        start_iter = checkpoint["iter"]

        if "height" in checkpoint.keys():
            hwf[0] = checkpoint["height"]
        if "width" in checkpoint.keys():
            hwf[1] = checkpoint["width"]
        if "focal_length" in checkpoint.keys():
            hwf[2] = checkpoint["focal_length"]

    nerf_network.model_coarse.eval()
    if nerf_network.model_fine:
        nerf_network.model_fine.eval()

    if not use_dataloader:
        poses = poses.float()

    # Create directory to save images to.
    os.makedirs(configargs.savedir, exist_ok=True)
    folder_coarse = os.path.join(configargs.savedir, "coarse")
    folder_fine = os.path.join(configargs.savedir, "fine")
    os.makedirs(folder_coarse, exist_ok=True)
    os.makedirs(folder_fine, exist_ok=True)

    if configargs.save_disparity_image:
        os.makedirs(os.path.join(folder_coarse, "disparity"), exist_ok=True)
        os.makedirs(os.path.join(folder_fine, "disparity"), exist_ok=True)

    # Evaluation loop
    times_per_image = []
    for i, img_idx in enumerate(i_test): #enumerate(tqdm(render_poses)):
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            # pose = pose[:3, :4]
            if not use_dataloader:
                img_target = images[img_idx]
                pose_target = poses[img_idx, :3, :4].to(device)
                if expressions is not None:
                    expressions_target = expressions[img_idx].to(device)
                else:
                    expressions_target = None

                if landmarks3d is not None:
                    landmarks3d_target = landmarks3d[img_idx].to(device)
                else:
                    landmarks3d_target = None
            else:
                data = test_dataset[img_idx]
                img_target = data["imgs"]
                pose_target = data["poses"][:3, :4].to(device)
                names = data["names"]

                if cfg.dataset.use_expression:
                    expressions_target = data["expressions"].to(device)
                else:
                    expressions_target = None

                if cfg.dataset.use_landmarks3d:
                    landmarks3d_target = data["landmarks3d"].to(device)
                else:
                    landmarks3d_target = None

            if cfg.dataset.refine_pose:
                pose_target = RefinePose()(nerf_network.refine_pose_params[img_idx], pose_target)

            if hasattr(cfg.dataset, "mask_face") and cfg.dataset.mask_face:
                out = face_seg_net.infer(img_target).astype(np.float32)
                # Mask out image and make background random
                img_target = img_target*out[:,:,None]+((1-out[:,:, None])*np.random.uniform(0,1,(1,1,3)))
            img_target = img_target.to(device)


            ray_origins, ray_directions = get_ray_bundle_nerface(H, W, focal, pose_target)
            nerf_network.img_idx = img_idx
            nerf_network.refine_pose = i/2e5 if cfg.dataset.refine_pose else None  # 2e5 following barf paper
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

                rgb_coarse, disp_coarse, _, rgb_fine, _, disp_fine ,weights_background_sample = run_one_iter_of_nerf(
                                H,
                                W,
                                focal,
                                nerf_network,
                                ray_origins,
                                ray_directions,
                                cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                    encode_ldmks_fn=encode_ldmks_fn,
                    encode_ldmks_dir_fn=encode_ldmks_dir_fn,
                    expressions=expressions_target,
                    # send all the background to generate the test image
                    background_prior=background_img.view(-1, 3) if cfg.dataset.fix_background else None,
                    landmarks3d=landmarks3d_target,
                    use_ldmks_dist=cfg.dataset.use_ldmks_dist,
                    cutoff_type=None if cfg.dataset.cutoff_type == "None" else cfg.dataset.cutoff_type,
                    embed_face_body=cfg.dataset.embed_face_body,
                    embed_face_body_separately=cfg.dataset.embed_face_body_separately,
                )
                target_ray_values = img_target

        times_per_image.append(time.time() - start)
        if configargs.savedir:
            # save coarse
            savefile = os.path.join(folder_coarse, names)
            imageio.imwrite(savefile, cast_to_image(rgb_coarse[..., :3], cfg.dataset.type.lower()))
            if configargs.save_disparity_image:
                savefile = os.path.join(folder_coarse, "disparity", names)
                imageio.imwrite(savefile, cast_to_disparity_image(disp_coarse))

            # save fine
            if rgb_fine is not None:
                savefile = os.path.join(folder_fine, names)
                imageio.imwrite(savefile, cast_to_image(rgb_fine[..., :3], cfg.dataset.type.lower()))
                if configargs.save_disparity_image:
                    savefile = os.path.join(folder_fine, "disparity", names)
                    imageio.imwrite(savefile, cast_to_disparity_image(disp_fine))

        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")


if __name__ == "__main__":
    main()
