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
    get_ray_bundle,
    load_blender_data,
    load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
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
    if cfg.dataset.type.lower() == "blender":
        # Load blender dataset
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif cfg.dataset.type.lower() == "face":
        images, poses, render_poses, hwf, i_split, expressions, landmarks3d, bboxs, names = load_nerface_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            load_expressions=cfg.dataset.use_expression,
            load_landmarks3d=cfg.dataset.use_landmarks3d,
        )

        if not cfg.dataset.use_expression:
            expressions = None

        # show_dirs(poses, cfg)

        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

    elif cfg.dataset.type.lower() == "llff":
        # Load LLFF dataset
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

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

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        num_encoding_fn_ldmks=cfg.models.coarse.num_encoding_fn_ldmks,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        include_input_ldmks=cfg.models.coarse.include_input_ldmks,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        use_expression=cfg.dataset.use_expression,
        use_landmarks3d=cfg.dataset.use_landmarks3d,
        use_appearance_code=cfg.dataset.use_appearance_code,
        use_deformation_code=cfg.dataset.use_deformation_code,
        num_train_images=len(i_train),
        landmarks3d_last=cfg.dataset.landmarks3d_last,
        encode_ldmks3d=cfg.dataset.encode_ldmks3d,
        embedding_vector_dim=cfg.dataset.embedding_vector_dim,
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            num_encoding_fn_ldmks=cfg.models.coarse.num_encoding_fn_ldmks,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            include_input_ldmks=cfg.models.coarse.include_input_ldmks,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            num_layers=cfg.models.coarse.num_layers,
            hidden_size=cfg.models.coarse.hidden_size,
            use_expression=cfg.dataset.use_expression,
            use_landmarks3d=cfg.dataset.use_landmarks3d,
            use_appearance_code=cfg.dataset.use_appearance_code,
            use_deformation_code=cfg.dataset.use_deformation_code,
            num_train_images=len(i_train),
            landmarks3d_last=cfg.dataset.landmarks3d_last,
            encode_ldmks3d=cfg.dataset.encode_ldmks3d,
            embedding_vector_dim=cfg.dataset.embedding_vector_dim,
        )
        model_fine.to(device)

    if cfg.dataset.fix_background:
        # based on nerface https://github.com/gafniguy/4D-Facial-Avatars/blob/989be64216df754a4a34f8f53d7a71af130b57d5/nerface_code/nerf-pytorch/train_transformed_rays.py#L160
        print("loading gt background to condition on")
        background_img = Image.open(os.path.join(cfg.dataset.basedir,'bg','00050.png'))
        background_img.thumbnail((H,W))
        background_img = torch.from_numpy(np.array(background_img).astype(np.float32)).to(device)
        background_img = background_img/255
        print("bg shape", background_img.shape)
        print("should be ", images[i_train][0].shape)
        assert background_img.shape == images[i_train][0].shape
    else:
        background_img = None

    checkpoint = torch.load(configargs.checkpoint)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )

    if "appearance_codes" in checkpoint and checkpoint["appearance_codes"] is not None:
        print("loading appearance codes from checkpoint")
        appearance_codes = torch.nn.Parameter(checkpoint['appearance_codes'].to(device))
    else:
        appearance_codes = None
    if "deformation_codes" in checkpoint and checkpoint["deformation_codes"] is not None:
        print("loading appearance codes from checkpoint")
        deformation_codes = torch.nn.Parameter(checkpoint['deformation_codes'].to(device))
    if "refine_pose_params" in checkpoint and checkpoint["refine_pose_params"] is not None:
        print("loading refine pose params from checkpoint")
        refine_pose_params = torch.nn.Parameter(checkpoint['refine_pose_params'].to(device))
    else:
        deformation_codes = None

    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

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
    for i, img_idx in enumerate(i_val): #enumerate(tqdm(render_poses)):
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            # pose = pose[:3, :4]
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            if expressions is not None:
                expressions_target = expressions[img_idx].to(device)
            else:
                expressions_target = None

            if landmarks3d is not None:
                landmarks3d_target = landmarks3d[img_idx].to(device)
            else:
                landmarks3d_target = None

            ray_origins, ray_directions = get_ray_bundle_nerface(H, W, focal, pose_target)
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ ,weights_background_sample = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                encode_ldmks_fn=encode_ldmks_fn,
                expressions=expressions_target,
                # send all the background to generate the test image
                background_prior=background_img.view(-1, 3) if cfg.dataset.fix_background else None,
                landmarks3d=landmarks3d_target,
                appearance_codes=appearance_codes[0].to(device) if cfg.dataset.use_appearance_code else None,  # it can be any from 0 to len(train_imgs) we chose 0 here
                deformation_codes=deformation_codes[0].to(device) if cfg.dataset.use_deformation_code else None,
                use_ldmks_dist=cfg.dataset.use_ldmks_dist,
                cutoff_type=None if cfg.dataset.cutoff_type == "None" else cfg.dataset.cutoff_type,
                embed_face_body=cfg.dataset.embed_face_body,
                embed_face_body_separately=cfg.dataset.embed_face_body_separately,
                refine_pose=1 if cfg.dataset.refine_pose else None,  # 2e5 following barf paper
                
            )
            target_ray_values = img_target
        times_per_image.append(time.time() - start)
        if configargs.savedir:
            # save coarse 
            savefile = os.path.join(folder_coarse, names[img_idx])
            imageio.imwrite(savefile, cast_to_image(rgb_coarse[..., :3], cfg.dataset.type.lower()))
            if configargs.save_disparity_image:
                savefile = os.path.join(folder_coarse, "disparity", names[img_idx])
                imageio.imwrite(savefile, cast_to_disparity_image(disp_coarse))

            # save fine
            if rgb_fine is not None:
                savefile = os.path.join(folder_fine, names[img_idx])
                imageio.imwrite(savefile, cast_to_image(rgb_fine[..., :3], cfg.dataset.type.lower()))
                if configargs.save_disparity_image:
                    savefile = os.path.join(folder_fine, "disparity", names[img_idx])
                    imageio.imwrite(savefile, cast_to_disparity_image(disp_fine))

        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")


if __name__ == "__main__":
    main()
