import torch

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field

from pytorch3d.ops.knn import knn_points
import matplotlib.pyplot as plt
import numpy as np


def get_pts_landmarks3d_dist(pts, landmarks3d):
    """pts: [N, 3] N is number of sampled piints of the whole image, not only a ray
       landmarks3d: [K, 3] K is the number of vertices or landmarks in the face mesh
    """
    dist = pts[:, None] - landmarks3d[None, :]
    norm = torch.linalg.norm(dist, axis=-1)  # [N, K, 3] = N, K
    dist = dist/norm[:, :, None]
    return norm, dist.reshape(pts.shape[0], -1) # [N,K], [N, K*3]
    #return dist.reshape(pts.shape[0], -1)  # [N, K, 3] -> [N, K*3]


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn, embedldmks_fn,
                expressions=None, landmarks3d=None, appearance_codes=None, deformation_codes=None,
                cutoff_type=None):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat, None, None)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat, None, None)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
    
    if landmarks3d is not None:
        # Get how for a sample point is from the K landmarks
        dist_pts_lndmks3d, dir_pts_ldmks3d = get_pts_landmarks3d_dist(pts_flat, landmarks3d)

        if cutoff_type is not None:
            tau = 100  # sharpness
            threshold_dist = 0.09  # threshold distance
            cutoff_w = 1-torch.sigmoid(tau*(dist_pts_lndmks3d-threshold_dist))
            # p_np = cutoff_w.min(axis=-1)[0].detach().cpu().numpy()
        else:
            cutoff_w = None
        embed_dists = embedldmks_fn(dist_pts_lndmks3d, cutoff_w, cutoff_type)
        
        embedded = torch.cat((embed_dists, dir_pts_ldmks3d, embedded), dim=-1)
    batches = get_minibatches(embedded, chunksize=chunksize)

    if expressions is None:
        preds = [network_fn(batch, appearance_codes=appearance_codes, deformation_codes=deformation_codes) for batch in batches]
    else:
        preds = [network_fn(batch, expressions, appearance_codes=appearance_codes, deformation_codes=deformation_codes) for batch in batches]
    
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    encode_ldmks_fn=None,
    expressions=None,
    background_prior=None,
    landmarks3d=None,
    appearance_codes=None,
    deformation_codes=None,
    use_ldmks_dist=False,
    cutoff_type=None,
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    # NOTE: here are the points that are sampled
    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
        encode_ldmks_fn,
        expressions=expressions,
        landmarks3d=landmarks3d,
        appearance_codes=appearance_codes,
        deformation_codes=deformation_codes,
        cutoff_type=cutoff_type,
    )
    if background_prior is not None:
        # make the last sample of the ray be equal to the background
        # NOTE: is this shperical?
        radiance_field[:, -1, :3] = background_prior

    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        background_prior=background_prior,
    )

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        if use_ldmks_dist:
            # get distance between sampled points pts and landmarks3d
            dist_pts2ldmks3d = knn_points(pts, landmarks3d[None].repeat(pts.shape[0],1,1), K=1)[0]
            alpha = 2000  # smoothing term. Controls how sharp the probability will be. The higher, the more sharp
            dist_pts2ldmks3d = torch.exp(-dist_pts2ldmks3d*alpha).squeeze()

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
            sample2ldmks_weights=dist_pts2ldmks3d[..., 1:-1] if use_ldmks_dist else None,
        )
        z_samples = z_samples.detach()

        ablation_plot_points = False
        if ablation_plot_points:                    
            plots = show_samples(5)
            pts_old = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
            pts_new = ro[..., None, :] + rd[..., None, :] * z_samples[..., :, None]
            plots.add_sample_weights(pts_old, weights)
            plots.add_sample_weights(pts_old, 
                                dist_pts2ldmks3d if use_ldmks_dist else weights)
            plots.add_sample_weights(pts_old, 
                            dist_pts2ldmks3d/dist_pts2ldmks3d.sum(axis=1, keepdims=True) + \
                                weights/weights.sum(axis=1, keepdims=True))
            plots.add_samples(pts_old, landmarks3d)
            plots.add_samples(pts_new, landmarks3d)
            plt.show()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            encode_ldmks_fn,
            expressions=expressions,
            landmarks3d=landmarks3d,
            appearance_codes=appearance_codes,
            deformation_codes=deformation_codes,
            cutoff_type=cutoff_type,
        )

        if background_prior is not None:
            # make the last sample of the ray be equal to the background
            # NOTE: is this shperical?
            radiance_field[:, -1, :3] = background_prior

        rgb_fine, disp_fine, acc_fine, weights, _ = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            background_prior=background_prior,
        )

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:, -1] # background weight


class show_samples:
    def __init__(self, n_figures):
        self.fig = plt.figure() #(figsize=(12, 8),)
        # ax = plt.figure().add_subplot(projection='3d')
        self.n_figures = n_figures
        self.count_figures = 0

    def add_subplot(self):
        self.count_figures += 1
        return self.fig.add_subplot(1, self.n_figures, self.count_figures, projection='3d')

    def add_samples(self, pts, landmarks3d):
        ax = self.add_subplot()
        ldmks3d_np = landmarks3d.detach().cpu().numpy()
        pts_np = pts.detach().cpu().reshape([-1, 3]).numpy()
        x, y, z = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2] 

        ax.plot(x,y,z, '.r')#, vmin=0, vmax=1)
        ax.plot(ldmks3d_np[:, 0], ldmks3d_np[:, 1], ldmks3d_np[:, 2], '.b')
        self.ax_properties(ax)


    def add_sample_weights(self, pts, weights):
        ax = self.add_subplot()

        # dist_pts2ldmks3d = dist_pts2ldmks3d/dist_pts2ldmks3d.sum(axis=1, keepdims=True)  #  remove, just to test how the prob for each ray will look like
        p_np = weights.detach().cpu().numpy()
        p_np = (p_np/p_np.sum(axis=1, keepdims=True)).flatten()

        mask = p_np > 0.01
        p_np = p_np[mask]

        pts_np = pts.detach().cpu().reshape([-1, 3]).numpy()
        pts_np = pts_np[mask]
        x, y, z = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2]


        scatter = ax.scatter(x,y,z, c=p_np, alpha=p_np, cmap=plt.cm.magma, vmin=0, vmax=0.5)#, vmin=0, vmax=1)        
        self.ax_properties(ax)
        plt.colorbar(scatter)

    def ax_properties(self, ax):
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_ylabel('$Z$')
        ax.set_xlim3d(-0.20, 0.20)
        ax.set_ylim3d(-0.30, 0.30)
        ax.set_zlim3d(-0.20, 0.20)


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    encode_ldmks_fn=None,
    expressions=None,
    background_prior=None,
    landmarks3d=None,
    appearance_codes=None,
    deformation_codes=None,
    use_ldmks_dist=False,
    cutoff_type=None,
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]] # to return last weight value (background)
    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    background_prior = get_minibatches(background_prior, chunksize=getattr(options.nerf, mode).chunksize) if\
        background_prior is not None else background_prior
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            mode=mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            encode_ldmks_fn=encode_ldmks_fn,
            expressions=expressions,
            background_prior=background_prior[i] if background_prior is not None else background_prior,
            landmarks3d=landmarks3d if landmarks3d is not None else None,
            appearance_codes=appearance_codes,
            deformation_codes=deformation_codes,
            use_ldmks_dist=use_ldmks_dist,
            cutoff_type=cutoff_type,
        )
        for i, batch in enumerate(batches)
    ]

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])
    return tuple(synthesized_images)
