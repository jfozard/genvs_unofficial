

"""
Models for generating novel views via a frustum-aligned latent NeRF
"""


import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nerf.network_multi import NeRFNetwork
from nerf.utils import render_multi_view


import random
from sd_pipeline_mv import SDPipeline
from k_diffusion.augmentation import KarrasDiffAugmentationPipeline

from dlv3.model import DeepLabV3Plus


def tv2(m):
    B, C, I, J = m.shape
    return ((m[:,:,1:,:] - m[:,:,:-1,:])**2).mean() + ((m[:,:,:,1:] - m[:,:,:,:-1])**2).mean()

def make_input_model(size=128, out_dim=3*48):
    """
    Function to make a dynamic U-Net model for input image.

    Parameters:
    size : int, optional
        The size of the input image.

    out_dim : int, optional
        The output dimensions.

    Returns:
    model : nn.Module
        The created model.
    """
    m = resnet34()
    m = nn.Sequential(*list(m.children())[:-2])
    model = DynamicUnet(m, out_dim, (size, size), norm_type=None)
    return model



class NerfDiffDLV3(nn.Module):
    """
    A NeRF-Diffusion model for processing multi-view input images.
    Uses DeepLabV3Plus as input unet.

    Parameters:
    input_size : int, optional
        The size of the input image.

    color_feat_dim : int, optional
        The dimension of color features.

    depth_size : int, optional
        The size of the depth field.

    a_prob : float, optional
        The probability for the KarrasDiffAugmentationPipeline.

    diff_augmentation_prob : float, optional
        The probability for applying diffusion augmentation.

    k_diffusion_config : str, optional
        The path to the configuration file for KPipeline.

    lambda_rgb_first : float, optional
        The loss weight for RGB rendering of NeRF for the first (projected) image.

    lambda_rgb_other : float, optional
        The loss weight for RGB rendering of NeRF from a novel view.

    lambda_opacity : float, optional
        The loss weight penalizing rendering opacity.

    lambda_diffusion : float, optional
        The loss weight for diffusion objective.

    lambda_depth: float, optional
        The loss weight for a term which pushes the mean depth towards the distance from the camera to the origin.

    lambda_depth_consistency: float, optional
        The loss weight measuring consistency between the depths estimated using the first (Q) views and the depth
        estimated from the final view.

    lambda_opacity_consistency: float, optional
        The loss weight measuring consistency between the opacities estimated using the first (Q) views and the opacity
        estimated from the final view.

    no_cond_prob : float, optional
        The probability of replacing the rendered NeRF with normally distributed noise,
        and adding a learnt vector to the timestep embedding of the denoising UNet.
    """    
    
    def __init__(self,
                 input_size=128,
                 train_diffusion_resolution=128,
                 color_feat_dim=16,
                 depth_size=64,
                 a_prob=0.1,
                 diff_augmentation_prob=0.2,
                 k_diffusion_config='k_configs/config_128_mid.json',
                 lambda_rgb_first=0.0,
                 lambda_rgb_other=1.0,
                 lambda_opacity=0.001,
                 lambda_diffusion=1.0,
                 lambda_depth=0.0,
                 lambda_depth_consistency=1.0,
                 lambda_opacity_consistency=1.0,
                 no_cond_prob=0.1):

        super().__init__()
        self.input_unet = DeepLabV3Plus() 
        self.nerf = NeRFNetwork(color_feat_dim=color_feat_dim) # Always render at 64x64 (but 16 channels)
        self.sd_pipeline = SDPipeline() 
        self.diff_aug = KarrasDiffAugmentationPipeline(a_prob = a_prob)
        self.diff_augmentation_prob = diff_augmentation_prob
        self.lambda_rgb_first = lambda_rgb_first # Loss weight for rgb rendering of NeRF for first (projected) image
        self.lambda_rgb_other = lambda_rgb_other # Loss weight for rgb rendering of NeRF from novel view
        self.lambda_opacity = lambda_opacity # Loss weight penalizing rendering opacity
        self.lambda_diffusion = lambda_diffusion # Loss weight for diffusion objective
        self.lambda_depth = lambda_depth # Loss weight for diffusion objective
        self.lambda_depth_consistency = lambda_depth_consistency # Loss weight for diffusion objective
        self.lambda_opacity_consistency = lambda_opacity_consistency # Loss weight for diffusion objective
        self.no_cond_prob= no_cond_prob # Probability of replacing the rendered NeRF with normally distributed noise,
                              # and adding a learnt vector to the timestep embedding of the
                              # denoising UNet.
        self.train_diffusion_resolution = train_diffusion_resolution

        print('lrf', self.lambda_rgb_first)

    def forward(self, data, depth_consistency=False):
        """
        Forward pass of the model.

        Parameters:
        data : dict
            A dictionary containing the input images, poses, camera parameters, etc.

        Returns:
        loss : float
            The combined loss from different stages of the model.

        loss_details : dict
            A dictionary containing the detailed loss from each stage.
        """
        img = data['imgs'].cuda()
        targets = 0.5*(img+1)
        
        B, V = targets.shape[:2]

        if self.train_diffusion_resolution == 64:
            # Downsample targets
            targets= F.interpolate(targets.view(B*V, *targets.shape[2:]), scale_factor = 0.5).view(B, V,3, 64, 64)
            
         
    
        nv = random.randint(1,2)

        # Augment input images with noise?
        
        imagenet_stats = (torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
        img_tp = (targets - (imagenet_stats[0][None,None,:,None,None]))/(imagenet_stats[1][None,None,:,None,None]) # 0 for orig xkpt

        B, V = img_tp.shape[:2]
        img_tp = img_tp.view(B*V,*img_tp.shape[2:])
        # Input augmentation (after normalization our images roughly range from -2 to 2, greater range than that in original paper).
        img_tp = img_tp + ((torch.rand([B*V], device=img_tp.device)>0.5)*torch.rand([B*V], device=img_tp.device))[:,None,None,None]*torch.randn_like(img_tp)
        triplanes_all = self.input_unet(torch.flip(img_tp[:,:],[2])) # Triplanes for the first view
        triplanes = triplanes_all.view(B, V, *triplanes_all.shape[1:])[:,:nv].contiguous()

        # Downsample targets

        poses = data['poses'].cuda()
        intrinsics = data['intrinsics']
        camera_k = data['camera_k'].cuda()
        camera_d = data['camera_d'].cuda()

        cameras  = (camera_k, camera_d, poses[:,:nv])

        Q = poses.shape[1]

        render_poses = poses #torch.cat([poses[:,:(Q-1)], poses[:,-1:]], dim=1)

        first_view, d1, o1 = render_multi_view(self.nerf, render_poses, intrinsics[0], triplanes, cameras)

        if self.train_diffusion_resolution == 128:
            first_view= F.interpolate(first_view.view(B*Q, *first_view.shape[2:]), scale_factor = 2)
            first_view = first_view.view(B, Q, *first_view.shape[1:])
            first_view_rgb = first_view[:, :, :3]
            d1 = F.interpolate(d1.view(B*Q,*d1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
            d1 = d1.view(B, Q, *d1.shape[1:])
            o1 = F.interpolate(o1.view(B*Q,*o1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
            o1 = o1.view(B, Q, *o1.shape[1:])
        else: # train diffusion model at resolution 64
            d1 = d1.expand(-1,-1,3,-1,-1)
            o1 = o1.expand(-1,-1,3,-1,-1)


        loss_rgb = self.lambda_rgb_other*((first_view_rgb - targets)**2).mean()

        # Penalize non-zero occupancy - attempt to remove floaters and other density that doesn't contribute to image
        loss_opacity =self.lambda_opacity*(o1.mean())

        # Try to ensure mean depth is close to difference from camera - try to avoid "floating filter"
        # near camera which might be being used to indicate view-angle dependent uncertainty
        
        loss_depth = self.lambda_depth*(o1*(d1/(o1+1e-6)-camera_d[:,None,None,None,None])**2).mean()
        
        # Denoise target image, guided by multi-channel rendering

        if self.lambda_diffusion:
            #cond_flag = 0 if random.random()>self.no_cond_prob else 1 # 10% chance of conditioning being replaced by noise
            cond_flag = torch.zeros((B,Q))<self.no_cond_prob
            
            mat, cond = self.diff_aug.get_mat_cond(first_view[:,0]) # Get the conditioning vector size
            cond = torch.zeros_like(cond)                            # Zero out conditioning vector
            target = img
            image = first_view

            loss_train_diffusion = self.lambda_diffusion*self.sd_pipeline.train_step(target, image, cond_flag=cond_flag, aug_cond=cond)

        else:
            loss_train_diffusion = torch.tensor([0.0], device=targets.device)
	# Combined loss

        loss_depth_consistency = torch.tensor([0.0]).cuda()
        loss_opacity_consistency = torch.tensor([0.0]).cuda()

        
        loss_details = { 'rgb': loss_rgb, 'opacity': loss_opacity, 'diffusion': loss_train_diffusion, 'depth': loss_depth,
                         'depth_consistency': loss_depth_consistency, 'opacity_consistency': loss_opacity_consistency }

        loss = loss_train_diffusion + loss_rgb + loss_opacity +loss_depth + loss_depth_consistency + loss_opacity_consistency

        
        return loss, loss_details

