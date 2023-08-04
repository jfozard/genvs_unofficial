import os
import random
import numpy as np

from packaging import version as pver
import torch


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
        inds = inds.expand([B, N])


        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = safe_normalize(directions)

    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def render_view(renderer, poses, intrinsics, triplanes, camera_k, camera_d, H=64, W=64):
    # sample a low-resolution but full image
    B, N = poses.shape[:2]

    rays = get_rays(poses, intrinsics, H, W, -1)
    rays_o = rays['rays_o'] # [B, N, 3]
    rays_d = rays['rays_d'] # [B, N, 3]

    bg_color = 1.0 #torch.rand((B * N, 3), device=rays_o.device) # pixel-wise random
    outputs = renderer.render(rays_o, rays_d, triplanes, camera_k, camera_d, staged=False, perturb=True, bg_color=bg_color, force_all_rays=True)
    pred_rgb = outputs['image'].reshape(B, H, W, renderer.color_feat_dim).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
    pred_depth = outputs['depth'].reshape(B, 1, H, W)
    pred_occ = outputs['weights_sum'].reshape(B, 1, H, W)
    
    return pred_rgb, pred_depth, pred_occ


def render_multi_view(renderer, poses, intrinsics, triplanes, cameras, H=64, W=64):
    # sample a low-resolution but full image
    B, Q = poses.shape[:2]

    rays = get_rays(poses.reshape(B*Q, *poses.shape[2:]), intrinsics, H, W, -1)
    rays_o = rays['rays_o'] # [BQ, N, 3]
    rays_d = rays['rays_d'] # [BQ, N, 3]

    rays_o = rays_o.view(B, Q, *rays_o.shape[1:])

    rays_d = rays_d.view(B, Q, *rays_d.shape[1:])

    N = rays_o.shape[2]

    bg_color = 1.0 #torch.rand((B * N, 3), device=rays_o.device) # pixel-wise random
    outputs = renderer.render(rays_o, rays_d, triplanes, cameras, staged=False, perturb=True, bg_color=bg_color, force_all_rays=True)
    pred_rgb = outputs['image'].reshape(B, Q, H, W, renderer.color_feat_dim).permute(0, 1, 4, 2, 3).contiguous() # [B, Q, 3, H, W]
    pred_depth = outputs['depth'].reshape(B, Q, 1, H, W)
    pred_occ = outputs['weights_sum'].reshape(B, Q, 1, H, W)
    
    return pred_rgb, pred_depth, pred_occ

