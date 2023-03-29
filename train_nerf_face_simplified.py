import argparse
import glob
import os
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import yaml
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_nerface_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf,
                  get_ray_bundle_nerface, RefinePose,
                  NerfBase, NerfFaceDataset, rescale_bbox)
from utils.losses import compute_losses, l2_nerf_loss
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

    if cfg.dataset.embed_face_body_separately:
        assert cfg.dataset.embed_face_body, "embed_face_body parameter has to be True"
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    images, poses, render_poses, hwf, i_split, expressions = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None

    use_dataloader = False
    if cfg.dataset.type.lower() == "face":
        images, poses, render_poses, hwf, i_split, expressions, landmarks3d, bboxs, names = load_nerface_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            load_expressions=cfg.dataset.use_expression,
            load_landmarks3d=cfg.dataset.use_landmarks3d,
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

    elif cfg.dataset.type.lower() == "face_dataloader":
        train_dataset = NerfFaceDataset(cfg.dataset.basedir, load_expressions = cfg.dataset.use_expression,
                                                        load_landmarks3d = cfg.dataset.use_landmarks3d,
                                    load_bbox = True, split = 'train', bbox_scale = 2.0)
        # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        eval_dataset = NerfFaceDataset(cfg.dataset.basedir, load_expressions = cfg.dataset.use_expression,
                                                        load_landmarks3d = cfg.dataset.use_landmarks3d,
                                    load_bbox = True, split = 'val', bbox_scale = 2.0)
        # eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
        i_train, i_val = (list(range(len(train_dataset))), list(range(len(eval_dataset))))
        use_dataloader = True
        H, W = train_dataset[0]["hw"].numpy()
        print(H,W)
        H, W = int(H), int(W)
        focal = train_dataset[0]["intrinsics"].numpy()
        print(focal)
    else:
        raise NotImplementedError

    if not cfg.dataset.use_expression:
        expressions = None
    # show_dirs(poses, cfg)


    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
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
    if use_amp:
        grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters, lr=cfg.optimizer.lr, eps=1e-15,)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters, lr=cfg.optimizer.lr)

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
        nerf_network.load_checkpoint(checkpoint, optimizer)

        start_iter = checkpoint["iter"]

    for i in trange(0, cfg.experiment.train_iters):
        nerf_network.model_coarse.train()
        if nerf_network.model_fine:
            nerf_network.model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        background_ray_values = None

        if i <= start_iter:
            # Trick to continue the random choice when loading the checkpoint
            # TODO: change for randomstate
            img_idx = np.random.choice(i_train)
            continue

        img_idx = np.random.choice(i_train)

        if not use_dataloader:
            img_target = images[img_idx] #.to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            bbox_img = bboxs[img_idx]

            if expressions is not None:
                expressions_target = expressions[img_idx].to(device)
            else:
                expressions_target = None

            if landmarks3d is not None:
                landmarks3d_target = landmarks3d[img_idx].to(device)
            else:
                landmarks3d_target = None
        else:
            data = train_dataset[img_idx]
            img_target = data["imgs"]
            pose_target = data["poses"][:3, :4].to(device)
            bbox_img = data["bbox"]

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
            # pose_target = compose_pair(pose_refine[img_idx], pose_target)

        if hasattr(cfg.dataset, "mask_face") and cfg.dataset.mask_face:
            from skimage.morphology import disk, binary_dilation
            big_bbox = bbox_img.float().numpy()
            big_bbox[:2] /= H
            big_bbox[2:] /= W
            big_bbox = rescale_bbox(big_bbox, 1.5)
            big_bbox[0:2] *= H  # top, left
            big_bbox[2:4] *= W  # right, bottom
            big_bbox = np.floor(big_bbox).astype(np.int)
            out_bbx = face_seg_net.infer(img_target[big_bbox[0]:big_bbox[1],big_bbox[2]:big_bbox[3]]).astype(np.float32)
            out = np.zeros([img_target.shape[0], img_target.shape[1]])
            out[big_bbox[0]:big_bbox[1],big_bbox[2]:big_bbox[3]] = out_bbx
            out = binary_dilation(out, disk(3, dtype=bool))
            # Mask out image and make background random
            img_target = img_target*out[:,:,None]+((1-out[:,:, None])*np.random.uniform(0,1,(1,1,3))).astype(np.float32)
        img_target = img_target.to(device)

        if "face" in cfg.dataset.type.lower():
            ray_origins, ray_directions = get_ray_bundle_nerface(H, W, focal, pose_target)
        else:
            raise NotImplementedError

        coords = torch.stack(
            meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
            dim=-1,
        )
        coords = coords.reshape((-1, 2))

        # Compute sampling probability map
        if cfg.dataset.sample_inside_bbox:
            probs = get_prob_map_bbox(bbox_img, H, W, p=0.9).reshape(-1)
        else:
            probs = None

        select_inds = np.random.choice(
            coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False,
            p=probs, # Sample more inside the bbox
        )
        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]

        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        background_ray_values = background_img[select_inds[:, 0], select_inds[:, 1], :] if cfg.dataset.fix_background else None

        nerf_network.img_idx = img_idx
        nerf_network.refine_pose = i/2e5 if cfg.dataset.refine_pose else None # 2e5 following barf paper
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            rgb_coarse, _, _, rgb_fine, _, _, weights_background_sample, weight_bce = run_one_iter_of_nerf(
                H,
                W,
                focal,
                nerf_network,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                encode_ldmks_fn=encode_ldmks_fn,
                encode_ldmks_dir_fn=encode_ldmks_dir_fn,
                expressions=expressions_target,
                background_prior=background_ray_values,
                landmarks3d=landmarks3d_target,
                use_ldmks_dist=cfg.dataset.use_ldmks_dist,
                cutoff_type=None if cfg.dataset.cutoff_type == "None" else cfg.dataset.cutoff_type,
                embed_face_body=cfg.dataset.embed_face_body,
                embed_face_body_separately=cfg.dataset.embed_face_body_separately,
                optimize_density=cfg.dataset.use_density_loss,
            )
            target_ray_values = target_s

            # Compute losses
            loss, losses_out = compute_losses(nerf_network, rgb_coarse, rgb_fine, target_ray_values, cfg)
            if cfg.dataset.use_density_loss:
                weight_bce = weight_bce.mean()
            else:
                weight_bce = 0
            loss = loss + 0.1*weight_bce
        if use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        psnr = mse2psnr(losses_out["loss_nerf"])

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
                + str(losses_out["loss_nerf"])
                + " PSNR: "
                + str(psnr)
                + " deformation: "
                + str(losses_out["loss_deformation_codes"] if cfg.optimizer.deformation_code else 0)
            )
        writer.add_scalar("train/loss", losses_out["loss_nerf"], i)
        writer.add_scalar("train/coarse_loss", losses_out["coarse_loss"], i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", losses_out["fine_loss"], i)
        writer.add_scalar("train/psnr", psnr, i)
        if cfg.optimizer.appearance_code and cfg.dataset.use_appearance_code:
            writer.add_scalar("train/l2_appearance_code", losses_out["loss_appearance_codes"], i)
        if cfg.optimizer.deformation_code and cfg.dataset.use_deformation_code:
            writer.add_scalar("train/l2_deformation_code", losses_out["loss_deformation_codes"], i)

        # Validation
        if (i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1):

            tqdm.write("[VAL] =======> Iter: " + str(i))
            nerf_network.model_coarse.eval()
            if nerf_network.model_fine:
                nerf_network.model_fine.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None

                loss, total_coarse_loss, total_fine_loss = 0., 0., 0.
                psnr_fine, psnr = 0., 0.
                # img_idx = np.random.choice(i_val)
                for img_idx in range(1): #i_val: Not worthy to test on all the 5 val images
                    img_idx = np.random.choice(i_val)

                    if not use_dataloader:
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
                    else:
                        data = eval_dataset[img_idx]
                        img_target = data["imgs"].to(device)
                        pose_target = data["poses"][:3, :4].to(device)

                        if cfg.dataset.use_expression:
                            expressions_target = data["expressions"].to(device)
                        else:
                            expressions_target = None

                        if cfg.dataset.use_landmarks3d:
                            landmarks3d_target = data["landmarks3d"].to(device)
                        else:
                            landmarks3d_target = None

                    if "face" in cfg.dataset.type.lower():
                        ray_origins, ray_directions = get_ray_bundle_nerface(H, W, focal, pose_target)
                    else:
                        raise NotImplementedError

                    nerf_network.img_idx = img_idx
                    nerf_network.refine_pose = i/2e5 if cfg.dataset.refine_pose else None  # 2e5 following barf paper
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        rgb_coarse, _, _, rgb_fine, _, _ ,weights_background_sample, _ = run_one_iter_of_nerf(
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

                        _, losses_out = l2_nerf_loss(rgb_coarse, rgb_fine, target_ray_values)

                loss += losses_out["loss_nerf"]
                total_coarse_loss += losses_out["coarse_loss"]
                total_fine_loss += losses_out["fine_loss"]

                # not worthy testing on all the 5 val images, 1 is fine
                # loss /= len(i_val)
                # total_coarse_loss /= len(i_val)
                # total_fine_loss /= len(i_val)

                psnr += mse2psnr(losses_out["loss_nerf"])
                writer.add_scalar("validation/loss", loss, i)
                writer.add_scalar("validation/coarse_loss", total_coarse_loss, i)
                if nerf_network.model_fine:
                    psnr_fine += mse2psnr(losses_out["fine_loss"])
                    writer.add_scalar("validation/fine_loss", total_fine_loss, i)
                    writer.add_scalar("validataion/psnr_fine", psnr_fine, i)
                writer.add_scalar("validataion/psnr", psnr, i)

                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", total_fine_loss, i)
                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                if cfg.dataset.fix_background:
                    writer.add_image(
                        "validation/background", cast_to_image(background_img[..., :3]), i)
                if cfg.dataset.fix_background or cfg.dataset.use_density_loss:
                    writer.add_image(
                        "validation/weights", (weights_background_sample.detach().cpu().numpy()), i, dataformats='HW')


                tqdm.write(
                    "Validation loss: "
                    + str(loss)
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": nerf_network.model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not nerf_network.model_fine
                else nerf_network.model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
                "appearance_codes": nerf_network.appearance_codes.data if cfg.dataset.use_appearance_code else None,
                "deformation_codes": nerf_network.deformation_codes.data if cfg.dataset.use_deformation_code else None,
                "refine_pose_params": nerf_network.refine_pose_params.data if cfg.dataset.refine_pose else None,
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
