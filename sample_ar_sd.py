
import os

from imageio import imwrite
import numpy as np
from tqdm import tqdm
import argparse

from einops import rearrange
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from NERFdataset_k import dataset

from nerf.utils import render_multi_view
import torch.nn as nn

from genvs_model_sd import NerfDiffDLV3

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb

import random
import os

# Options - augment inputs, diffaugment on outputs
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=sampler)
    
    return sampler, dataloader
    
def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


def generate_circle_poses(ref_pose, radius, n_poses=120, d_th=-5, d_phi=-5):

    print('ref pose', ref_pose, ref_pose.shape)

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        
        rotation_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        rotation_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        #cam_to_world = trans_t(radius)
        
        

        rotation = lambda th: np.array([
            [np.cos(th), -np.sin(th), 0, 0],
            [np.sin(th), np.cos(th), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        cam_to_world = rotation_phi(phi / 180. * np.pi) #@ cam_to_world
        cam_to_world = rotation(theta) @ cam_to_world
        #cam_to_world = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        #                        dtype=np.float32) @ cam_to_world
        return cam_to_world

    spheric_cams = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_cams += [spheric_pose(th, np.sin(2*th)*0, radius) @ ref_pose ]

    return np.stack(spheric_cams, 0)


@torch.no_grad()
def sample_sphere_ar(model, data, source_view_idx,  steps=50, cfg=2.0, churn=0.0, n_poses=200, sample_view_batch=2, n_itm=1):
    # model - NerfDiff model
    # data - dataset batch 
    # source_view_idx - list of view indicies used to generate NeRF
    # sample_view_batch - number of novel views to generate at the same time
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)
    
    targets = targets[:,source_view_idx]

    generated_views = [targets[:, source_view_idx].cpu()]
    generated_poses = [data['poses'][:, source_view_idx].cpu()]

    poses = data['poses']
    intrinsics = data['intrinsics']
    camera_k = data['camera_k'].cuda()
    camera_d = data['camera_d'].cuda()

    render_output_views = []
    render_rgb_views = []

    render_output_depth = []
    render_output_opacities = []

    ref_pose = poses[:,0]
    sphere_poses = generate_circle_poses(data['poses'][0, source_view_idx[0]].numpy(), camera_d[0].cpu(), n_poses=n_poses)
  
    poses = torch.tensor(sphere_poses[None]).cuda()

    imagenet_stats = (torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
    


    nposes = poses.shape[1]

    random.seed(123)
    for q in range(1, nposes):
    	    
        # render a single view at a time autoregressively
        render_poses = poses[:,q:q+1]    

        QQ = render_poses.shape[1]

        if q>2:
            idx = [0] + random.sample(range(1, q-1), min(4, q-2)) + [len(generated_views)-1]
        elif q==2:
            idx = [0,1]
        else:
            idx = [0]

        selected_targets = torch.cat([generated_views[v] for v in idx], dim=1).cuda()
        selected_poses = torch.cat([generated_poses[v] for v in idx], dim=1).cuda()

        img_tp = (selected_targets - (imagenet_stats[0][None,None,:,None,None]))/(imagenet_stats[1][None,None,:,None,None]).contiguous()

        B, V = img_tp.shape[:2]
        img_tp = img_tp.view(B*V,*img_tp.shape[2:])
        triplanes = model.module.input_unet(torch.flip(img_tp[:,:],[2])) # Triplanes for the first view
        triplanes = triplanes.view(B, V, *triplanes.shape[1:])

        cameras  = (camera_k, camera_d, selected_poses)

        first_view, d1, o1 = render_multi_view(model.module.nerf, render_poses, intrinsics[0], triplanes, cameras)
        print('first_view', first_view.shape, B, QQ)

        first_view= F.interpolate(first_view.view(B*QQ, *first_view.shape[2:]), scale_factor = 2)
        first_view = first_view.view(B, QQ, *first_view.shape[1:])
        first_view_rgb = first_view[:, :, :3]

        print('first view shape', first_view.shape)

        
        d1 = F.interpolate(d1.view(B*QQ,*d1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
        d1 = d1.view(B, QQ, *d1.shape[1:])
        o1 = F.interpolate(o1.view(B*QQ,*o1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
        o1 = o1.view(B, QQ, *o1.shape[1:])

        render_output_depth.append(d1.cpu())
        render_output_opacities.append(o1.cpu())
        render_rgb_views.append(first_view_rgb.cpu())

        samples1 = model.module.sd_pipeline.sample_k(first_view.view(B*QQ, *first_view.shape[2:]), sampling_timesteps=steps, cfg=cfg, churn=churn)
        print('samples range', samples1.min(), samples1.max())
        samples1 = torch.tensor([np.array(t).transpose((2,0,1))/255.0 for t in samples1], dtype=torch.float32)
        samples1 = samples1.view(B, QQ, *samples1.shape[1:])
        print('s1 shape', samples1.shape)
        render_output_views.append(samples1.cpu())

        generated_poses.append(render_poses.cpu())
        generated_views.append(samples1.cpu())

    render_output_views = torch.cat(render_output_views, dim=1)
    render_output_depth = torch.cat(render_output_depth, dim=1)
    render_output_opacities = torch.cat(render_output_opacities, dim=1)
    render_rgb_views = torch.cat(render_rgb_views, dim=1)

    return targets.cpu(), render_output_views, render_rgb_views, render_output_depth, render_output_opacities




    # Generate poses

    # From mipnerf https://github.com/google/mipnerf
def generate_spherical_cam_to_world(radius, n_poses=120, d_th=-5, d_phi=-5):
    """
    Generate a 360 degree spherical path for rendering
    ref: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
    ref: https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_cams: (n_poses, 3, 4) the cam to world transformation matrix of a circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        rotation_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        rotation_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        cam_to_world = trans_t(radius)
        cam_to_world = rotation_phi(phi / 180. * np.pi) @ cam_to_world
        cam_to_world = rotation_theta(theta) @ cam_to_world
        cam_to_world = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                                dtype=np.float32) @ cam_to_world
        return cam_to_world

    spheric_cams = []
    for th, phi in zip(np.linspace(0, 4 * np.pi, n_poses + 1)[:-1], np.linspace(10, -60, n_poses + 1)[:-1]):
        spheric_cams += [spheric_pose(th, phi, radius)]

    return np.stack(spheric_cams, 0)



def convert_and_make_grid(views):
    def convert(x):
        return x.numpy().transpose(1,2,0)

    views = list(map(convert, views))

    output = np.concatenate( (np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)))

    return output
    


def sample_images(rank, world_size, transfer="", cond_views=1, prefix="cars", steps=50, n=1, cfg=1, churn=0.0, offset=0, n_poses=10, use_wandb = False):

    setup(rank, world_size)

    if use_wandb and rank==0:
        wandb.init(
        entity=None,
        project="genvs",
        job_type="train",
	    )
    
        wandb.define_metric("*", step_metric="train/step")
        wandb.define_metric("train/step", step_metric="walltime")


    # ------------ Init
    step = 0
    image_size = 128
    batch_size = 1
    n_samples = n



    d = dataset('test', imgsize=image_size, nimg=4, normalize_first_view=False, first_view=0, idx_offset=offset)
    
    sampler, loader = prepare(rank, world_size, d, batch_size=batch_size)

    # Model setting
    model = NerfDiffDLV3().cuda()
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], output_device=rank)

    use_amp=False

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load saved model if defined
    if transfer !="":
             # inner_model.proj_in.weight

        print('resume from: ', transfer)

        #ckpt = torch.load(os.path.join(transfer, 'large-k-multi-latest.pt'))#, map_location=map_location)

        ckpt = torch.load(transfer)
        model.module.load_state_dict(ckpt['model'])


        del ckpt



#    model.eval()
    
    pbar = tqdm(loader)

    cond_view_list = list(range(cond_views))

    for step, data in enumerate(pbar):
        if step==n_samples:
            cleanup()
            return

        original_views, render_output_views, render_rgb_views, render_output_depth, render_output_opacities = sample_sphere_ar(model, data, cond_view_list, steps=steps, cfg=cfg, churn=churn, n_poses=n_poses)


        conditioning_views = original_views
        output = np.concatenate([v.numpy().transpose(1,2,0) for v in conditioning_views[0,:]])
        output = (255*np.clip(output,0,1)).astype(np.uint8)

        imwrite(f'output_view_sd/{prefix}-conditioning-ar-{cond_views}-{step:06d}.png', output)

        
        for k in range(render_output_views.shape[1]):
 
            

            output = convert_and_make_grid((render_output_views[0,k], render_rgb_views[0,k], render_output_depth[0,k], render_output_opacities[0,k]))
            
            output = (255*np.clip(output,0,1)).astype(np.uint8)
            imwrite(f'output_view_sd/{prefix}-sample-ar-{cond_views}-{step:06d}-{k}.png', output)

        del original_views, render_output_views, render_rgb_views, render_output_depth, render_output_opacities

            
    cleanup()


output_path = 'output_view_sd/'
os.makedirs(output_path, exist_ok=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer',type=str, default="")
    parser.add_argument('--cond_views',type=int, default=1)
    parser.add_argument('--prefix',type=str, default="cars")
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cfg', type=float, default=1)
    parser.add_argument('--churn', type=float, default=0.0)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--n_poses', type=int, default=10)
    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    mp.spawn(sample_images, args=(world_size,args.transfer, args.cond_views, args.prefix, args.steps, args.n, args.cfg, args.churn, args.offset, args.n_poses), nprocs=world_size, join=True)
