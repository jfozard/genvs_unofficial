
"""
GeNVS - Neural Radiance Fields (NeRF) model for view synthesis with the 
support of distributed training across multiple GPUs. 
This is done with the help of PyTorch's distributed training package.
The NeRF model aims to synthesize novel views of a scene given a sparse 
set of input views.
"""
from omegaconf import DictConfig, OmegaConf
import hydra

import os

from imageio import imwrite
import numpy as np
from tqdm import tqdm
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import  AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from NERFdataset_k import dataset
from genvs_model_sd_mv import NerfDiffDLV3
from nerf.utils import render_multi_view
import torch.nn as nn


import torch.nn as nn
from collections import defaultdict


import wandb

# Different tasks  
#   1) just train RGB
#   2) import pre-trained diffusion model (unconditional)
#   3) train at 64x64
#   4) train at 128x128, importing 64x64 model

# Options - augment inputs, diffaugment on outputs

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from k_pipeline import KPipeline


from k_diffusion.augmentation import KarrasDiffAugmentationPipeline

output_dir = 'output_sd_multi/'
os.makedirs(output_dir, exist_ok=True)

def setup(rank, world_size):
    """
    Sets up the distributed environment for PyTorch.

    Parameters:
    rank (int): Rank of the current process.
    world_size (int): Total number of processes.
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    """
    Prepares a DataLoader and a DistributedSampler for training.

    Parameters:
    rank (int): Rank of the current process.
    world_size (int): Total number of processes.
    dataset (torch.utils.data.Dataset): Dataset to be prepared for distributed training.
    batch_size (int): How many samples per batch to load. Default is 32.
    pin_memory (bool): Whether to have the data returned in pinned (page-locked) memory. Default is False.
    num_workers (int): How many worker processes to use for data loading. Default is 0.

    Returns:
    sampler (DistributedSampler), dataloader (DataLoader): The prepared sampler and dataloader.
    """
    # Note shuffle
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=sampler)
    
    return sampler, dataloader
    
def cleanup():
    """
    Cleans up the distributed environment.
    """

    dist.destroy_process_group()

@torch.no_grad()
def sample(model, data, nv):
    """
    Performs sampling from the model during inference.

    Parameters:
    model (nn.Module): The model to perform sampling from.
    data (dict): A dictionary containing the data to sample.
    nv (int): Number of views to sample.

    Returns:
    A tuple containing targets, first view, depth maps, and samples.
    """
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)
    
    B, V = targets.shape[:2]

    imagenet_stats = (torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
    img_tp = (targets - (imagenet_stats[0][None,None,:,None,None]))/(imagenet_stats[1][None,None,:,None,None]) 

    B, V = img_tp.shape[:2]
    img_tp = img_tp.view(B*V,*img_tp.shape[2:])
    triplanes = model.module.input_unet(torch.flip(img_tp[:,:],[2])) # Triplanes for the first view
    triplanes = triplanes.view(B, V, *triplanes.shape[1:])[:,:nv].contiguous()

    # Downsample targets

    poses = data['poses'].cuda()
    intrinsics = data['intrinsics']
    camera_k = data['camera_k'].cuda()
    camera_d = data['camera_d'].cuda()

    cameras  = (camera_k, camera_d, poses[:,:nv])

    Q = 4

    render_poses = poses[:,:(Q)]    

    first_view, d1, o1 = render_multi_view(model.module.nerf, render_poses, intrinsics[0], triplanes, cameras)
    print('first_view', first_view.min(), first_view.max())
    first_view= F.interpolate(first_view.view(B*Q, *first_view.shape[2:]), scale_factor = 2)
    first_view = first_view.view(B, Q, *first_view.shape[1:])
    first_view_rgb = first_view[:, :, :3]


    d1 = F.interpolate(d1.view(B*Q,*d1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
    d1 = d1.view(B, Q, *d1.shape[1:])
    o1 = F.interpolate(o1.view(B*Q,*o1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
    o1 = o1.view(B, Q, *o1.shape[1:])


    d1 = d1 - d1.min()
    d1 = d1/d1.max()

    samples1 = model.module.sd_pipeline.sample_k(first_view.view(B*Q, *first_view.shape[2:]), cfg=3)

    print('samples1', samples1.shape)

#    return targets[:,0], first_view_rgb[:,0], d1[:,0], o1[:,0], targets[:,1], first_view_rgb[:,1], d1[:,1], o1[:,1], samples1 #torch.cat((samples1, samples2), dim=0)
    return targets[:,:Q], first_view_rgb, d1, o1, samples1 #torch.cat((samples1, samples2), dim=0)


def stack_img(v):
    v = v.cpu().clip(0,1).numpy().transpose(0,1,3,4,2)
    v = v.reshape(v.shape[0], v.shape[1]*v.shape[2], v.shape[3], v.shape[4])
    return v

def train(rank, world_size, cfg):
    """
    Trains a Neural Radiance Fields (NeRF) model in a distributed manner.

    Parameters:
    rank (int): Rank of the current process.
    world_size (int): Total number of processes.
    cfg (dict): Configuration dictionary
    """
    setup(rank, world_size)

    print(os.listdir())

    transfer = cfg["transfer"] #(str): Path to a pretrained model if transfer learning is to be performed. Default is "".
    use_wandb = cfg["use_wandb"] #(bool): Whether to use Weights & Biases for logging and visualization. Default is False.

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
    num_epochs = cfg.max_epochs
    image_size = cfg.image_size
    batch_size = cfg.batch_size
    acc_steps = cfg.gradient_accumulation_steps
    n_sample = cfg.sample_every_n_steps
    combine = cfg.combine

    epochs_plot_loss = cfg.epochs_checkpoint
    epochs_plot_loss2 = cfg.epochs_checkpoint_save
    load_optim = cfg.load_optim_state_from_checkpoint

    if use_wandb and rank==0:
        wandb.config.update(
	        {
            "train_batch_size":batch_size,
            }
        )

    d = dataset('train', imgsize=image_size)
    
    sampler, loader = prepare(rank, world_size, d, batch_size=batch_size)

    # Model setting
    nerf_diff_params = cfg.nerf_diff
    model = NerfDiffDLV3(**nerf_diff_params).cuda()
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], output_device=rank)


    use_amp=cfg.use_amp
    #optimizer = AdamW([{'params':model.module.input_unet.parameters(), 'lr':1e-4}, {'params':model.module.nerf.parameters(), 'lr':2e-4}, {'params':model.module.ddpm_pipeline.parameters(), 'lr':1e-5}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    #optimizer = AdamW([{'params':model.module.input_unet.parameters(), 'lr':cfg.lr_in}, {'params':model.module.nerf.parameters(), 'lr':cfg.lr_nerf}, {'params':model.module.sd_pipeline.parameters(), 'lr':cfg.lr_diff}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    optimizer = AdamW([{'params':model.module.sd_pipeline.parameters(), 'lr':cfg.lr_diff}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
#    optimizer = AdamW([{'params':model.module.input_unet.parameters(), 'lr':1e-4}, {'params':model.module.nerf.parameters(), 'lr':2e-4}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load saved model if defined
    if transfer == "":
        step = 0
        start_epoch = 0
    elif transfer !="":
        if combine:

            print('Combine weights from: ', transfer, 'and a k-diffusion checkpoint, adding extra (zero-initialized) channels to k-diffusion input', )

            transfer_filename = cfg['transfer_filename']
            ckpt = torch.load(os.path.join(transfer, transfer_filename))
            model.module.load_state_dict(ckpt['model'], strict=False)

            sd_ckpt = torch.load(cfg['sd_checkpoint_path'])
            sd_weights = sd_ckpt['model']

            print([d for d in sd_weights])
            print([d[0] for d in model.module.named_parameters()])
            with torch.no_grad(): 
                key = 'sd_pipeline.controlnet.controlnet_cond_embedding.conv_in.weight'
                print(sd_weights[key].shape, dict(model.module.named_parameters())[key].shape)
                sd_weights[key] = torch.cat((sd_weights[key], torch.zeros((320,13,3,3), dtype=sd_weights[key].dtype, device=sd_weights[key].device )), dim=1)
                sd_weights[key].requires_grad_(True)
                print('modded_weights')
                
                model.module.load_state_dict(sd_weights, strict=False)
                print('loaded_weights')

            """
            transfer_filename = cfg['transfer_filename']
            ckpt = torch.load(os.path.join(transfer, transfer_filename))#, map_location=map_location)

            model.module.load_state_dict(ckpt['model'])
            

            step = ckpt['step']
            start_epoch = ckpt['epoch']+1   
            """
            step = 0
            start_epoch = 0

            del ckpt, sd_weights
        else:
            print('resume from: ', transfer)
            transfer_filename = cfg['transfer_filename']
            ckpt = torch.load(os.path.join(transfer, transfer_filename))


            model.module.load_state_dict(ckpt['model'], strict=False)
            if load_optim:
                optimizer.load_state_dict(ckpt['optim'])

            step = ckpt['step']
            start_epoch = ckpt['epoch']+1   

            del ckpt




    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=1, eta_min=1e-8, last_epoch=-1)

    t00 = t0 = time.monotonic()
    # Training loop
    for e in range(start_epoch, num_epochs):

        print(f'starting epoch {e}')
        
        model.train()
        
        print('LR', [p['lr'] for p in optimizer.param_groups])
        lt = time.time()

        #sampler.set_epoch(e)

        # For each sample in the dataset
        pbar = tqdm(loader)

        epoch_loss = 0.0
        running_loss = 0.0
        running_loss_rgb = 0.0
        epoch_loss_details = defaultdict(float)

        for data in pbar:

            if (step+1)%acc_steps:
                with model.no_sync():
                    with torch.cuda.amp.autocast(enabled=use_amp):
               
            # Forward and loss compute
                        loss, loss_details = model(data)

                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
               
            # Forward and loss compute
                    loss, loss_details = model(data)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss
            for d in loss_details:
                epoch_loss_details[d] += loss_details[d].item()

            running_loss += loss.item()



            if step % n_sample == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'loss : {running_loss_rgb/n_sample} {running_loss/n_sample} {lr}')
                if use_wandb:
                    wandb.log(
                        {
                            "walltime": time.monotonic() - t00,
                            "train/step": step,
                            "train/epoch": e,
                            **{f"train/{k}": v for k, v in loss_details.items()},
            			}
                    )

            
                running_loss = 0.0
                running_loss_rgb = 0.0

                formatted_images = []
                for vv in [1,2]:
                    img1, v1, d1, o1, samples = sample(model, data, vv)

                    img1 = stack_img(img1)
                    v1 = stack_img(v1)
                    d1 = stack_img(d1)
                    o1 = stack_img(o1)
                    
                    print('img1', img1.shape, 'samples', samples.shape)
                    for k in range(len(img1)):
#                        output = np.concatenate((np.concatenate((img1.cpu().detach().numpy()[k].transpose(1,2,0), v1.cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(d1,0,1).cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(o1,0,1).cpu().detach().numpy()[k].transpose(1,2,0)), axis=1),
#                                            np.concatenate((img2.cpu().detach().numpy()[k].transpose(1,2,0), v2.cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(d2,0,1).cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(o2,0,1).cpu().detach().numpy()[k].transpose(1,2,0)), axis=1)), axis=0)
                        output = np.concatenate((img1[k], v1[k], d1[k], o1[k]), axis=1)

                        output = (255*np.clip(output,0,1)).astype(np.uint8)
                        imwrite(f'{output_dir}/full-mv-{step:06d}-{vv}-{k}.png', output)
                        if use_wandb:
                            formatted_images.append(wandb.Image(output, caption=f"{vv}-{k}"))

                    for i in range(len(samples)):
                        s = np.asarray(samples[i]) #(255*(samples[i].clip(0,1)).cpu().detach().numpy().transpose(1,2,0)).astype(np.uint8)
                        imwrite(f'{output_dir}/sample-mv-{step:06d}-{vv}-{i}.png', s)
                        if use_wandb:
                            formatted_images.append(wandb.Image(s, caption=f"sample-{vv}-{i}"))


                del img1, v1, d1, o1, samples

                if use_wandb:
                    wandb.log({"validation": formatted_images})

            step += 1
        scheduler.step()
            
        print('loss epoch', epoch_loss / len(loader))
        epoch_loss_details = {k:v/len(loader) for k,v in epoch_loss_details.items()}
        print('loss epoch', epoch_loss_details)

        if use_wandb and rank==0:
            wandb.log(
                 {

                            **{f"train/epoch-{k}": v for k, v in epoch_loss_details.items()},
            	  }
                )

        # Epoch checkpoint save
        if e % epochs_plot_loss == 0 and rank==0:
            torch.save({'optim':optimizer.state_dict(), 'model':model.module.state_dict(), 'step':step, 'epoch':e}, f"{output_dir}/large-k-multi-latest.pt")
        if e % epochs_plot_loss2 == 0 and rank==0:
            torch.save({'optim':optimizer.state_dict(), 'model':model.module.state_dict(), 'step':step, 'epoch':e}, f"{output_dir}/large-k-multi-epoch-{e}.pt")
            if use_wandb:
                wandb.save(f"{output_dir}/sd-large-k-multi-epoch-{e}.pt")


            

    
@hydra.main(version_base="1.2", config_path="config", config_name="config_sd_multi")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    n_gpus = torch.cuda.device_count()
    #assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":

    main()
