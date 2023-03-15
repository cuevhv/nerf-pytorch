import numpy as np
import torch
import os
from typing import Tuple


class NerfBase(object):
    def __init__(self, train_size, device) -> None:
        self.train_size = train_size  # len(i_train)
        self.model_coarse = None
        self.model_fine = None
        self.deformation_codes = None
        self.appearance_codes = None
        self.refine_pose_params = None
        self.optimizer = None
        self.device = device
        self.img_idx = None
        self.refine_pose = None


    def create_nerf_network(self, cfg, models):
        self.model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        num_encoding_fn_ldmks=cfg.models.coarse.num_encoding_fn_ldmks,
        num_encoding_fn_dir_ldmks= cfg.models.coarse.num_encoding_fn_dir_ldmks \
                            if hasattr(cfg.nerf, "encode_ldmks_direction_fn") else 0,
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
        num_train_images=self.train_size,
        landmarks3d_last=cfg.dataset.landmarks3d_last,
        encode_ldmks3d=cfg.dataset.encode_ldmks3d,
        embedding_vector_dim=cfg.dataset.embedding_vector_dim,
        n_landmarks=68+8 if "iris" in cfg.dataset.basedir else 68,
        )
        self.model_coarse.to(self.device)

        # If a fine-resolution model is specified, initialize it.
        if hasattr(cfg.models, "fine"):
            self.model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            num_encoding_fn_ldmks=cfg.models.coarse.num_encoding_fn_ldmks,
            num_encoding_fn_dir_ldmks= cfg.models.coarse.num_encoding_fn_dir_ldmks \
                                        if hasattr(cfg.nerf, "encode_ldmks_direction_fn") else 0,
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
            num_train_images=self.train_size,
            landmarks3d_last=cfg.dataset.landmarks3d_last,
            encode_ldmks3d=cfg.dataset.encode_ldmks3d,
            embedding_vector_dim=cfg.dataset.embedding_vector_dim,
            n_landmarks=68+8 if "iris" in cfg.dataset.basedir else 68,
            )

            self.model_fine.to(self.device)


    def init_network(self, cfg, models):
        self.create_nerf_network(cfg, models)
        trainable_parameters = list(self.model_coarse.parameters())
        if self.model_fine is not None:
            trainable_parameters += list(self.model_fine.parameters())

        # Adding learnable codes
        if cfg.dataset.use_appearance_code:
            self.appearance_codes = self.create_learnable_codes((self.train_size, 32))
            trainable_parameters.append(self.appearance_codes)
        if cfg.dataset.use_deformation_code:
            self.deformation_codes = self.create_learnable_codes((self.train_size, cfg.dataset.embedding_vector_dim))
            trainable_parameters.append(self.deformation_codes)
        if cfg.dataset.refine_pose:
            self.refine_pose_params = self.create_learnable_codes((self.train_size, 6))
            trainable_parameters.append(self.refine_pose_params)

        # return all trainable parameters
        return trainable_parameters

    def create_learnable_codes(self, code_shape: Tuple[int, int]):
        print("initialized latent codes with shape %d X %d" % (code_shape[0], code_shape[1]))
        return torch.zeros(code_shape[0], code_shape[1], device=self.device).requires_grad_()


    def load_checkpoint(self, checkpoint, optim):
        self.model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            self.model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if "appearance_codes" in checkpoint and checkpoint["appearance_codes"] is not None:
            print("loading appearance codes from checkpoint")
            self.appearance_codes = torch.nn.Parameter(checkpoint['appearance_codes'].to(self.device))
        if "deformation_codes" in checkpoint and checkpoint["deformation_codes"] is not None:
            print("loading deformation codes from checkpoint")
            self.deformation_codes = torch.nn.Parameter(checkpoint['deformation_codes'].to(self.device))
        if "refine_pose_params" in checkpoint and checkpoint["refine_pose_params"] is not None:
            print("loading refine pose params from checkpoint")
            self.refine_pose_params = torch.nn.Parameter(checkpoint['refine_pose_params'].to(self.device))

        if optim is not None:
            print("loading optimizer checkpoint")
            optim.load_state_dict(checkpoint["optimizer_state_dict"])


    def slice_code(self, learnable_code):
        if learnable_code is not None:
            return learnable_code[self.img_idx].to(self.device)
