

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import safe_normalize

def sample_pdf(bins, weights, n_samples, det=False):

    #print('bins, weights', bins.shape, weights.shape)
    B, Q, N, T = bins.shape
    bins = bins.view(B*Q*N, T)
    weights = weights.view(B*Q*N, -1)

    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    samples = samples.view(B,Q,N,-1)

    return samples

@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=min_near)

    return near, far


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self, bound=1.0, min_near=0.5):
        super().__init__()

        self.bound = bound 
        self.min_near = min_near


        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-self.bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()



    def run(self, rays_o, rays_d, triplanes, cameras, num_steps=64,
            upsample_steps=64,
            light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        
        # rays_o, rays_d: [B, Q, N, 3]
        # bg_color: [BN, 3] ? in range [0, 1]
        # return: image: [B, Q, N, 3], depth: [B, Q, N]
        

        B, Q, N = rays_o.shape[:3]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous()#.view(-1, 3)
        rays_d = rays_d.contiguous()#.view(-1, 3)

        N = rays_o.shape[2] 
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='cube', min_near=self.min_near)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)


        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]

        #print('xyzs', xyzs.shape)

        # Now map xyzs into component NeRF frames

#        if self.frustrum:
#            xyzs[:,:,0:2] = camera_k*xyzs[:,:,0:2]/torch.clip(camera_d - xyzs[:,:,2:3], 1e-6)

        xyzs_clip = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.


        # query SDF and RGB
        density_outputs = self.density(xyzs, triplanes, cameras)

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            #print(k, v.shape)
            density_outputs[k] = v.view(B, Q, N, num_steps, -1)
            #print(k, density_outputs[k].shape)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]
                #weights = alphas * torch.cumprod(alphas, dim=-1)[..., :-1] # [N, T]
                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[..., 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]


            # only forward new points to save computation
            #new_density_outputs = self.density(new_xyzs.reshape(-1, 3), triplanes, cameras)
            new_density_outputs = self.density(new_xyzs, triplanes, cameras)
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(B, Q, N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=-1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=-1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=-2) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=-2, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                #print(k, density_outputs[k].shape,new_density_outputs[k].shape)
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=-2)
                density_outputs[k] = torch.gather(tmp_output, dim=-2, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
#        print('m', mask.shape, alphas.shape)

        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        
        #print('rays_d, xyzs', rays_d.shape, xyzs.shape)
        dirs = rays_d.unsqueeze(3).expand_as(xyzs)
        #for k, v in density_outputs.items():
        #    density_outputs[k] = v.view(-1, v.shape[-1])

        sigmas, rgbs, normals = self(xyzs, dirs, triplanes, cameras)
        #print('rgbs', rgbs.shape)
        rgbs = rgbs.view(B, Q, N, -1, self.color_feat_dim) # [N, T+t, 3]

        if normals is not None:
            # orientation loss
            normals = normals.view(B, Q, N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.sum(-1).mean()

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) #mask.mean(dim=-1) # [N]

        
        # calculate depth 
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            bg_color = self.background(rays_d) # [N, 3]
        elif self.bg_radius ==0:
            bg_color = 1

        bg_depth = 2.0
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        #depth = (depth+0.1*bg_depth)/(weights_sum+0.1) #+ (1 - weights_sum) * bg_depth

        image = image.view(*prefix, self.color_feat_dim)
        depth = depth.view(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights'] = weights
        results['weights_sum'] = weights_sum


        return results



    def render(self, rays_o, rays_d, triplanes, cameras, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]


        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, self.color_feat_dim), device=device)
            weights_sum = torch.empty((B, N), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], triplanes, cameras, **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    weights_sum[b:b+1, head:tail] = results_['weights_sum']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum

        else:
            results = _run(rays_o, rays_d, triplanes, cameras, **kwargs)

        return results
