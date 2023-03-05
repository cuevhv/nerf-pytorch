import torch
import numpy as np


def l2_nerf_loss(rgb_coarse, rgb_fine, target_ray_values):
    coarse_loss = torch.nn.functional.mse_loss(
        rgb_coarse[..., :3], target_ray_values[..., :3]
    )
    fine_loss = None
    if rgb_fine is not None:
        fine_loss = torch.nn.functional.mse_loss(
            rgb_fine[..., :3], target_ray_values[..., :3]
        )
    # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])

    loss_nerf = coarse_loss + (fine_loss if fine_loss is not None else 0.0)

    losses_out = {'loss_nerf': loss_nerf.item(), 'coarse_loss': coarse_loss.item(), 'fine_loss': fine_loss.item() if rgb_fine is not None else 0.}
    return loss_nerf, losses_out


def compute_losses(nerf_network, rgb_coarse, rgb_fine, target_ray_values, cfg):
    img_idx = nerf_network.img_idx
    loss_nerf, loss_appearance_codes, loss_deformation_codes = 0.0, 0.0, 0.0

    loss_nerf, losses_out = l2_nerf_loss(rgb_coarse, rgb_fine, target_ray_values)

    if cfg.optimizer.appearance_code and cfg.dataset.use_appearance_code:
        loss_appearance_codes = torch.linalg.norm(nerf_network.appearance_codes[img_idx])
        # loss = loss + 0.005*loss_appearance_codes
        losses_out['loss_appearance_codes'] = loss_appearance_codes.item()

    if cfg.optimizer.deformation_code and cfg.dataset.use_deformation_code:
        if cfg.dataset.embed_face_body:
            loss_deformation_codes = torch.linalg.norm(nerf_network.deformation_codes[img_idx, :cfg.dataset.embedding_vector_dim//2]) + \
                                            torch.linalg.norm(nerf_network.deformation_codes[img_idx, cfg.dataset.embedding_vector_dim//2:])
        else:
            loss_deformation_codes = torch.linalg.norm(nerf_network.deformation_codes[img_idx])
        losses_out['loss_deformation_codes'] = loss_deformation_codes.item()

        # loss = loss + 0.005*loss_deformation_codes

    loss = loss_nerf + 0.005*loss_appearance_codes + 0.005*loss_deformation_codes
    return loss, losses_out