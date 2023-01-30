import torch


class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(
            self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[..., : self.xyz_encoding_dims], x[..., self.xyz_encoding_dims :]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)


class PaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(PaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
        for i in range(1, 8):
            if i == 4:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = xyz #self.relu(self.layers_xyz[0](xyz))
        for i in range(8):
            if i == 4:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class FlexibleNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(FlexibleNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class FlexibleNeRFaceModel(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        hidden_size: int = 128,
        skip_connect_every: int = 4,
        num_encoding_fn_xyz: int = 6,
        num_encoding_fn_dir: int = 4,
        num_encoding_fn_ldmks: int = 4,
        include_input_xyz: bool = True,
        include_input_dir: bool = True,
        include_input_ldmks: bool = True,
        use_viewdirs: bool = True,
        use_expression: bool = True,
        use_landmarks3d: bool = True,
        use_appearance_code: bool =True,
        use_deformation_code: bool =True,
        num_train_images: int = 0,
        embedding_vector_dim: int = 32,  # based on nerface
        landmarks3d_last: bool = False,
    ):
        super(FlexibleNeRFaceModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_input_ldmks = 1 if include_input_ldmks else 0
        
        # TODO: change expression and lanrmsrks3d value to depend on cfg
        include_expresion = 50 if use_expression else 0
        include_landmarks3d = 68 if use_landmarks3d else 0

        self.landmarks3d_last = landmarks3d_last

        # add appearance code
        self.use_appearance_code = use_appearance_code
        self.use_deformation_code = use_deformation_code
        # if use_appearance_code:
            # self.appearance_codes = torch.nn.Embedding(num_embeddings=num_train_images, 
            #                                            embedding_dim=embedding_vector_dim)


        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expresion
        self.dim_landmarks3d = include_input_ldmks*include_landmarks3d + 2 * include_landmarks3d * num_encoding_fn_ldmks + include_landmarks3d*3
        self.dim_appearance_codes = embedding_vector_dim if use_appearance_code else 0
        self.dim_deformation_codes = embedding_vector_dim if use_deformation_code else 0 


        self.skip_connect_every = skip_connect_every
        self.use_landmarks3d = use_landmarks3d
        if not use_viewdirs:
            self.dim_dir = 0

        # dim of the first input group to predict the density
        input_density_dim = self.dim_xyz + self.dim_expression + self.dim_deformation_codes
        if not landmarks3d_last:
            input_density_dim += self.dim_landmarks3d

        self.layer1 = torch.nn.Linear(input_density_dim, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(input_density_dim + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            # dim of the second input group to predict the color
            input_color_dim = self.dim_dir + self.dim_appearance_codes
            if landmarks3d_last:
                input_color_dim += self.dim_landmarks3d

            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(input_color_dim + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

    def forward(self, x, expression=None, appearance_codes=None, deformation_codes=None):
        if self.use_landmarks3d:
            if not self.landmarks3d_last:
                xyz, dirs = x[..., : self.dim_landmarks3d+self.dim_xyz], x[..., self.dim_landmarks3d+self.dim_xyz :]
            else:
                xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        elif self.use_viewdirs:
            xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        
        if self.dim_expression:
            # The face has the same expression in all the pixels of the image
            # NOTE: maybe input expression only on the pixels where the face is?
            expressions = (expression * 1 / 3).repeat(xyz.shape[0], 1)
            xyz = torch.cat((xyz, expressions), dim=1)

        if self.use_deformation_code:
                deformation_codes = deformation_codes.repeat(xyz.shape[0], 1)
                xyz = torch.cat((xyz, deformation_codes), dim=1)
        
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, dirs), dim=-1)

            if self.use_appearance_code:
                appearance_codes = appearance_codes.repeat(xyz.shape[0], 1)
                x = torch.cat((x, appearance_codes), dim=1)
                
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)

class FaceNerfPaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        num_encoding_fn_ldmks=4,
        include_input_xyz=True,
        include_input_dir=True,
        include_input_ldmks=True,
        use_viewdirs=True,
        use_expression=True,
        use_landmarks3d: bool = True,
        use_appearance_code: bool =True,
        use_deformation_code: bool = True,
        num_train_images: int = 0,
        embedding_vector_dim=32,
        landmarks3d_last=False,
        encode_ldmks3d=False,

    ):
        super(FaceNerfPaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_input_ldmks = 1 if include_input_ldmks else 0

        include_expression = 50 if use_expression else 0
        include_landmarks3d = 68 if use_landmarks3d else 0

        self.landmarks3d_last = landmarks3d_last

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_landmarks3d = include_input_ldmks*include_landmarks3d + 2 * include_landmarks3d * num_encoding_fn_ldmks + include_landmarks3d*3
        self.dim_full_landmarks3d = self.dim_landmarks3d

        self.encode_ldmks3d = encode_ldmks3d
        if self.encode_ldmks3d:
            self.layers_ldmks3d_enc = torch.nn.ModuleList()
            self.layers_ldmks3d_enc.append(torch.nn.Linear(self.dim_landmarks3d+self.dim_xyz, 128))
            self.layers_ldmks3d_enc.append(torch.nn.Linear(128, 128))
            self.layers_ldmks3d_enc.append(torch.nn.Linear(128, self.dim_xyz))
            torch.nn.init.uniform_(self.layers_ldmks3d_enc[-1].weight, a=-1e-4, b=1e-4)
            self.dim_landmarks3d = 0

        # add appearance code
        self.use_appearance_code = use_appearance_code
        self.use_deformation_code = use_deformation_code
        self.dim_appearance_codes = embedding_vector_dim if use_appearance_code else 0
        self.dim_deformation_codes = embedding_vector_dim if use_deformation_code else 0
        # self.dim_latent_code = embedding_vector_dim

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.use_landmarks3d = use_landmarks3d

        # dim of the first input group to predict the density
        input_density_dim = self.dim_xyz + self.dim_expression + self.dim_deformation_codes
        if not landmarks3d_last:
            input_density_dim += self.dim_landmarks3d

        self.layers_xyz.append(torch.nn.Linear(input_density_dim, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(input_density_dim + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()

        # dim of the second input group to predict the color
        input_color_dim = self.dim_dir + self.dim_appearance_codes
        if landmarks3d_last:
            input_color_dim += self.dim_landmarks3d

        self.layers_dir.append(torch.nn.Linear(256 + input_color_dim, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expression=None, appearance_codes=None, deformation_codes=None, **kwargs):
        if self.use_landmarks3d:
            if not self.landmarks3d_last:
                xyz, dirs = x[..., : self.dim_full_landmarks3d+self.dim_xyz], x[..., self.dim_full_landmarks3d+self.dim_xyz :]
                if self.encode_ldmks3d:
                    xyz_pts = xyz[..., :self.dim_xyz]
                    for i in range(len(self.layers_ldmks3d_enc)):
                        xyz = self.layers_ldmks3d_enc[i](xyz)
                        if i < len(self.layers_ldmks3d_enc)-1:
                            xyz = self.relu(xyz)
                    xyz = xyz + xyz_pts
            else:
                xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        elif self.use_viewdirs:
            xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        
        x = xyz#self.relu(self.layers_xyz[0](xyz))
        initial = xyz

        # appearance_codes = appearance_codes.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expression * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((initial, expr_encoding), dim=1)
            # initial = torch.cat((xyz, expr_encoding, appearance_codes), dim=1)
        if self.use_deformation_code:
            deformation_codes = deformation_codes.repeat(xyz.shape[0], 1)
            initial = torch.cat((initial, deformation_codes), dim=1)
        x = initial
                   
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)

        if self.use_viewdirs:
            if self.use_appearance_code:
                appearance_codes = appearance_codes.repeat(xyz.shape[0], 1)
                x = self.layers_dir[0](torch.cat((feat, dirs, appearance_codes), -1))
            else:
                x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class translation_field(torch.nn.Module):
    def __init__(self, mlp_depth, mlp_dim, hidden_init, output_init, skips) -> None:
        super(translation_field).__init__()
        self.skips = skips
        self.mlp_layers = torch.nn.ModuleList()
        
        self.mlp_layers.append(torch.nn.Linear(mlp_dim, 256))
        for i in range(1, mlp_depth):
            if i in skips:
                self.mlp_layers.append(torch.nn.Linear(mlp_dim + 256, 256))
            else:
                self.mlp_layers.append(torch.nn.Linear(256, 256))
    def forward(self):
        return None
