import argparse
import glob
import os
import time

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_nerface_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf,
                  get_ray_bundle_nerface)

# from utils.viewer import show_dirs

def get_prob_map_bbox(bbox, H, W, p=0.9):
    probs = np.zeros((H,W))
    probs.fill(1-p)  # assigning low probability to all pixels
    probs[bbox[0]:bbox[1],bbox[2]:bbox[3]] = p  # assigning high prob to pixels inside the bbox
    probs = (1/probs.sum()) * probs  # make all the prob pixels sum to 1
    return probs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split, expressions = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                testskip=cfg.dataset.testskip,
            )

            # show_dirs(poses, cfg)

            i_train, i_val, i_test = i_split
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

        if cfg.dataset.type.lower() == "face":
            images, poses, render_poses, hwf, i_split, expressions, landmarks3d, bboxs = load_nerface_data(
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
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])


        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        use_expression=cfg.dataset.use_expression,
        use_landmarks3d=cfg.dataset.use_landmarks3d,
        use_appearance_code=cfg.dataset.use_appearance_code,
        use_deformation_code=cfg.dataset.use_deformation_code,
        num_train_images=len(i_train),
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            num_layers=cfg.models.coarse.num_layers,
            hidden_size=cfg.models.coarse.hidden_size,
            use_expression=cfg.dataset.use_expression,
            use_landmarks3d=cfg.dataset.use_landmarks3d,
            use_appearance_code=cfg.dataset.use_appearance_code,
            use_deformation_code=cfg.dataset.use_deformation_code,
            num_train_images=len(i_train),
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
    

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    
    if cfg.dataset.use_appearance_code:
        # appearance_codes = torch.zeros(len(i_train), 32, device=device).requires_grad_()
        appearance_codes = (torch.randn(len(i_train), 32, device=device)*0.1).requires_grad_()
        print("initialized latent codes with shape %d X %d" % (appearance_codes.shape[0], appearance_codes.shape[1]))
        # appearance_codes.requires_grad = True
        trainable_parameters.append(appearance_codes)
    
    if cfg.dataset.use_deformation_code:
        deformation_codes = (torch.randn(len(i_train), 32, device=device)*0.1).requires_grad_()
        print("initialized latent codes with shape %d X %d" % (deformation_codes.shape[0], deformation_codes.shape[1]))
        trainable_parameters.append(deformation_codes)
    
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if checkpoint["appearance_codes"] is not None:
            print("loading appearance codes from checkpoint")
            appearance_codes = torch.nn.Parameter(checkpoint['appearance_codes'].to(device))
        if checkpoint["deformation_codes"] is not None:
            print("loading appearance codes from checkpoint")
            deformation_codes = torch.nn.Parameter(checkpoint['deformation_codes'].to(device))
        
        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # Prepare importance sampling maps
    # NOTE: DO WE HAVE TO PRECOMPUTE THEM? I THINK NO
    # ray_importance_sampling_maps = []
    # print("computing boundix boxes probability maps")
    # for i in i_train:
    #     bbox = bboxs[i]
    #     probs = get_prob_map_bbox(bbox, H, W, p=0.9)
    #     ray_importance_sampling_maps.append(probs.reshape(-1))
        
    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        background_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            rgb_coarse, _, _, rgb_fine, _, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions=expressions
            )
        else:
            img_idx = np.random.choice(i_train)
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

            if cfg.dataset.type.lower() == "face":
                ray_origins, ray_directions = get_ray_bundle_nerface(H, W, focal, pose_target)
            else:
                ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))

            # Compute sampling probability map
            if cfg.dataset.sample_inside_bbox:
                probs = get_prob_map_bbox(bboxs[img_idx], H, W, p=0.9).reshape(-1)
            else:
                probs = None

            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False, 
                p=probs, # Sample more inside the bbox
            )
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            
            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            background_ray_values = background_img[select_inds[:, 0], select_inds[:, 1], :] if cfg.dataset.fix_background else None

            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _, weights_background_sample = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions=expressions_target,
                background_prior=background_ray_values,
                landmarks3d=landmarks3d_target,
                appearance_codes=appearance_codes[img_idx].to(device) if cfg.dataset.use_appearance_code else None,
                deformation_codes=deformation_codes[img_idx].to(device) if cfg.dataset.use_deformation_code else None,
            )
            target_ray_values = target_s

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        loss_nerf, loss_appearance_codes, loss_appearance_codes = 0.0, 0.0, 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss
        loss_nerf = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        # loss = loss_nerf

        if cfg.optimizer.appearance_code and cfg.dataset.use_appearance_code:
            loss_appearance_codes = torch.linalg.norm(appearance_codes[img_idx])
            # loss = loss + 0.005*loss_appearance_codes
        
        if cfg.optimizer.deformation_code and cfg.dataset.use_deformation_code:
            loss_deformation_codes = torch.linalg.norm(deformation_codes[img_idx])
            # loss = loss + 0.005*loss_deformation_codes
            
        loss = loss_nerf + 0.005*loss_appearance_codes + 0.005*loss_deformation_codes

        loss.backward()
        psnr = mse2psnr(loss_nerf.item())
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss_nerf.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss_nerf.item(), i)
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)
        if cfg.optimizer.appearance_code and cfg.dataset.use_appearance_code:
            writer.add_scalar("train/l2_appearance_code", loss_appearance_codes.item(), i)
        if cfg.optimizer.deformation_code and cfg.dataset.use_deformation_code:
            writer.add_scalar("train/l2_deformation_code", loss_deformation_codes.item(), i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_fine.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _, _ = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        expressions=expressions,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    loss, total_coarse_loss, total_fine_loss = 0., 0., 0. 
                    # img_idx = np.random.choice(i_val)
                    for img_idx in range(1): #i_val: Not worthy to test on all the 5 val images
                        img_idx = np.random.choice(i_val)
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

                        if cfg.dataset.type.lower() == "face":
                            ray_origins, ray_directions = get_ray_bundle_nerface(H, W, focal, pose_target)
                        else:
                            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)

                        rgb_coarse, _, _, rgb_fine, _, _ ,weights_background_sample = run_one_iter_of_nerf(
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
                            expressions=expressions_target,
                            # send all the background to generate the test image
                            background_prior=background_img.view(-1, 3) if cfg.dataset.fix_background else None,
                            landmarks3d=landmarks3d_target,
                            appearance_codes=appearance_codes[0].to(device) if cfg.dataset.use_appearance_code else None,  # it can be any from 0 to len(train_imgs) we chose 0 here
                            deformation_codes=deformation_codes[0].to(device) if cfg.dataset.use_deformation_code else None,
                        )
                        target_ray_values = img_target
                    coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                    # loss, fine_loss = 0.0, 0.0
                    fine_loss = 0.0
                    if rgb_fine is not None:
                        fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                        # loss = fine_loss
                    # else:
                        # loss = coarse_loss
                    loss += coarse_loss + fine_loss
                    total_coarse_loss += coarse_loss
                    total_fine_loss += fine_loss

                # not worthy testing on all the 5 val images, 1 is fine
                # loss /= len(i_val)
                # total_coarse_loss /= len(i_val) 
                # total_fine_loss /= len(i_val)

                psnr = mse2psnr(loss.item())
                psnr_fine = mse2psnr(total_fine_loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", total_coarse_loss.item(), i)
                writer.add_scalar("validation/fine_loss", total_fine_loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                writer.add_scalar("validataion/psnr_fine", psnr_fine, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", total_fine_loss.item(), i)
                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                if cfg.dataset.fix_background:
                    writer.add_image(
                        "validation/background", cast_to_image(background_img[..., :3]), i)
                    writer.add_image(
                        "validation/weights", (weights_background_sample.detach().cpu().numpy()), i, dataformats='HW')


                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
                "appearance_codes": appearance_codes.data if cfg.dataset.use_appearance_code else None,
                "deformation_codes": deformation_codes.data if cfg.dataset.use_deformation_code else None,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    main()
