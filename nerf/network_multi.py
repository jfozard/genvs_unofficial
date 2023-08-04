import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import trunc_exp
from .renderer_multi import NeRFRenderer

import numpy as np
from einops import einsum

from .utils import safe_normalize

import itertools

def repeat_el(lst, n):
    return list(itertools.chain.from_iterable(itertools.repeat(x, n) for x in lst))

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation='softplus', norm=False, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.norm = nn.LayerNorm(dim_out) if norm else nn.Identity()

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        if norm:
            with torch.no_grad():
                nn.init.zeros_(self.dense.bias)

        if activation == 'softplus':
            self.activation = nn.Softplus() #nn.ReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.norm(out)
        out = self.activation(out)

        return out    

    
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, activation='softplus', bias=True, bias_out=True, block=BasicBlock, norm=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []

        if num_layers==1:
            l = nn.Linear(self.dim_in, self.dim_out, bias=bias_out)
            if norm and bias_out:
                with torch.no_grad():
                    nn.init.zeros_(l.bias)
            net.append(l)
        else:            
            for l in range(num_layers):
                if l == 0:
                    net.append(BasicBlock(self.dim_in, self.dim_hidden, activation=activation, norm=norm, bias=bias))
                elif l != num_layers - 1:
                    net.append(block(self.dim_hidden, self.dim_hidden, activation=activation, norm=norm, bias=bias))
                else:
                    l = nn.Linear(self.dim_hidden, self.dim_out, bias=bias_out)
                    if norm and bias_out:
                        with torch.no_grad():
                            nn.init.zeros_(l.bias)
                    net.append(l)
                
            
        self.net = nn.ModuleList(net)
        
    
    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 n_resolutions=1,
                 n_resolutions_normal=1,
                 resolution=128,
                 resolution_z=64,
                 sigma_color_rank=16,
                 normal_rank=32,
                 color_feat_dim=3,
                 normal_feat_dim=3, 
                 num_layers=2, 
                 hidden_dim=64, 
                 ):
        super().__init__()

        self.resolution = resolution
        self.resolution_z = resolution_z
        
        # vector-matrix decomposition
        self.sigma_color_rank = sigma_color_rank
        self.normal_rank = normal_rank
        self.color_feat_dim = color_feat_dim
        self.normal_feat_dim = normal_feat_dim

        self.frustrum = True

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.sigma_color_net = MLP(self.sigma_color_rank, 1+self.color_feat_dim, hidden_dim, num_layers, bias=True)

        self.density_activation =  F.softplus

        self.bg_net = None
        self.bg_radius = 0 #None


    def get_sigma_color_feat(self, x, triplanes):

        B, V, Q, N, S = x.shape[0:5]  # BVQN3
        d = self.resolution_z

        x = x.view(B*V, Q*N*S, 3)

        B, V, C, H, W = triplanes.shape
        triplanes = triplanes.view(B*V, C//d, d, H, W)

        mat_coord = x.view(B*V,1,1,-1,3)
        mat_feat = F.grid_sample(triplanes, mat_coord, align_corners=True)
        mat_feat = mat_feat.view(B, V, -1, Q, N*S) # [B*V, C, 1, 1, Q*N*S] -> [ B, V, C, Q, N*S]

        sigma_color_feat = mat_feat.permute((0,1,3,4,2)) # [ B V C Q NS ] -> [B V Q N*S C]

        return sigma_color_feat

    def get_cam_coords(self, x, cameras):
        # normalize to [-1, 1] inside aabb_train
        #x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1
                
        camera_k, camera_d, poses = cameras

        # Poses are R, T, cam -> world
        R = poses[:,:,:3,:3]  # [B,V,3,3]
        T = poses[:,:,:3,3]   # [B,V,3]
        camera_d = T.norm(dim=2)
        # x has shape [B,Q,N,3]

        # x  [B, Q, N, S, 3]
    
        x = einsum(R, x[:,None,:,:,:,:] - T[:,:,None,None,None,:], 'b v i j, b v q n s i -> b v q n s j')

        x[...,2] += camera_d[:, :, None, None, None]

        if self.frustrum:
            # print('camera_k', camera_k.shape, camera_d.shape, x[...,0:2].shape)
            # This is wrong
            x[..., 0:2] = camera_k[:,None,None,None,None,None]*x[...,0:2]/torch.clip(camera_d[:,:,None,None,None,None] - x[...,2:3], 1e-6)
            x = torch.min(torch.max(x, self.aabb_train[:3]), self.aabb_train[3:])

        x_cam = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1
        return x_cam
    
    def forward(self, x, d, triplanes, cameras):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        # Convert points into camera coordinates

        x_cam = self.get_cam_coords(x, cameras)

        # sigma
        sigma_color_feat = self.get_sigma_color_feat(x_cam, triplanes) # [B V Q N*S C]
        sigma_color_feat = self.sigma_color_net(sigma_color_feat) # [B V Q N*S 17]

        sigma_color_feat = sigma_color_feat.mean(dim=1) # Average over V==projected views
        # Split this into density (sigma) and colour features

        sigma_feat = sigma_color_feat[...,0]
        color_feat = sigma_color_feat[...,1:]
        
        # Exp or softplus activation for density
        sigma = self.density_activation(sigma_feat)

        # sigmoid activation for rgb
        albedo = torch.sigmoid(color_feat)

        normal = None
        color = albedo
           
        return sigma, color, normal

      
    def density(self, x, triplanes, cameras):
        # x: [N, 3], in [-bound, bound]

        x_cam = self.get_cam_coords(x, cameras)
        
        sigma_color_feat = self.get_sigma_color_feat(x_cam, triplanes)
        sigma_color_feat = self.sigma_color_net(sigma_color_feat).mean(dim=1)


        sigma_feat = sigma_color_feat[...,0]

        # Exp or softplus activation for density
        sigma = self.density_activation(sigma_feat)
        color_feat = sigma_color_feat[...,1:]
        
        # sigmoid activation for rgb
        albedo = torch.sigmoid(color_feat)
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos = self.density((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dx_neg = self.density((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dy_pos = self.density((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dy_neg = self.density((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dz_pos = self.density((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dz_neg = self.density((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal


    def normal_fd(self, x):

        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)

        return normal


