
import os

from imageio import imwrite
import numpy as np
from tqdm import tqdm
import time
import argparse

from einops import rearrange
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from NERFdataset_k import dataset
import torch.nn as nn

import torch.nn as nn
from collections import defaultdict
from torchvision.transforms.functional import gaussian_blur

import random

from sd_pipeline import SDPipeline




class NerfDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.sd_pipeline = SDPipeline(cond_channels=3, unlock_up_blocks=True, cfg=2.0)


def losses(model, data):
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)
 
    #targets = gaussian_blur(targets[:,1], 15, (2,6))
    targets = torch.stack([gaussian_blur(targets[i,1], 21, random.randint(1,9)) for i in range(targets.shape[0])], dim=0)

    bsz = img.shape[0]

    #print(img[:,1].shape, targets[:,1].shape)

    cond = torch.rand((bsz,))<0.5
    
    input_ids = model.sd_pipeline.tokenize_captions(['a car']*bsz).cuda()

    loss_train_sd = model.sd_pipeline.train_step(img[:,1], targets.detach(), cond_flag=cond)

    loss = loss_train_sd
    
    return loss

def sample(model, data, unconditional=False):
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)
    bsz = img.shape[0]
    targets = torch.stack([gaussian_blur(targets[i,1], 21, random.randint(1,9)) for i in range(targets.shape[0])], dim=0)

    input_ids = ['a car']*bsz


    samples = model.sd_pipeline.sample(targets, unconditional=unconditional)


    return targets, samples

def train(rank, world_size, transfer=""):


    # ------------ Init
    step = 0
    num_epochs = 801
    image_size = 128
    batch_size = 32
    acc_steps = 16


    n_workers = 6
    epochs_plot_loss = 50


    ns = 500
    
    d = dataset('train', imgsize=image_size)
    
    loader = DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers)
    
    # Model setting
    model = NerfDiff().to('cuda:0')

    print([(m[0], m[1].requires_grad) for m in model.sd_pipeline.named_parameters()])

    use_amp=False
    optimizer = AdamW([{'params':model.sd_pipeline.parameters(), 'lr':5e-6}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    #optimizer = AdamW([{'params':model.sd_pipeline.parameters(), 'lr':1e-5}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    #optimizer = AdamW([{'params':model.sd_pipeline.parameters(), 'lr':1e-4}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load saved model if defined
    if transfer == "":
        step = 0
        start_epoch = 0
    else:
        print('transfering from: ', transfer)
        
        # Mapped ckpt loading

        ckpt = torch.load(transfer)#, map_location=map_location)

        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])

        step = ckpt['step']
        start_epoch = ckpt['epoch']+1
    # Training loop
    for e in range(start_epoch, num_epochs):

        print(f'starting epoch {e}')
        
        model.train()
        
        lt = time.time()

        # For each sample in the dataset
        pbar = tqdm(loader)

        epoch_loss = 0.0
        running_loss = 0.0

        for data in pbar:

            with torch.cuda.amp.autocast(enabled=use_amp):
               
            # Forward and loss compute
                loss = losses(model, data)/acc_steps

            scaler.scale(loss).backward()

            if (step+1)%acc_steps ==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss

            running_loss += loss.item()

            if step % ns == 0:

                pbar.set_description(f'loss : {running_loss/ns}')
                running_loss = 0.0


                targets, samples = sample(model, data)

#                output = np.concatenate((np.concatenate((0.5*(img1.cpu().detach().numpy()[0].transpose(1,2,0)+1), v1.cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(d1,0,1).cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(o1,0,1).cpu().detach().numpy()[0].transpose(1,2,0)), axis=1),
#                                          np.concatenate((0.5*(img2.cpu().detach().numpy()[0].transpose(1,2,0)+1), v2.cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(d2,0,1).cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(o2,0,1).cpu().detach().numpy()[0].transpose(1,2,0)), axis=1)), axis=0)
                
                for k in range(len(samples)):
                    output = np.concatenate(((255*targets.cpu().detach().numpy()[k%len(targets)].transpose(1,2,0)).astype(np.uint8), samples[k]), axis=1)


                    imwrite(f'copy-blur-{step:06d}-{k}.png', (output).astype(np.uint8))

                targets, samples = sample(model, data, unconditional=True)

#                output = np.concatenate((np.concatenate((0.5*(img1.cpu().detach().numpy()[0].transpose(1,2,0)+1), v1.cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(d1,0,1).cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(o1,0,1).cpu().detach().numpy()[0].transpose(1,2,0)), axis=1),
#                                          np.concatenate((0.5*(img2.cpu().detach().numpy()[0].transpose(1,2,0)+1), v2.cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(d2,0,1).cpu().detach().numpy()[0].transpose(1,2,0), torch.clip(o2,0,1).cpu().detach().numpy()[0].transpose(1,2,0)), axis=1)), axis=0)
                
                for k in range(len(samples)):
                    output = np.concatenate(((255*targets.cpu().detach().numpy()[k%len(targets)].transpose(1,2,0)).astype(np.uint8), samples[k]), axis=1)


                    imwrite(f'copy-blur-uc-{step:06d}-{k}.png', (output).astype(np.uint8))


            step += 1
            
        print('loss epoch', epoch_loss / len(loader))

        # Epoch checkpoint save
        #if (e+1) % epochs_plot_loss == 0:
        #    torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, "new-latest.pt")
        #    torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, f"new-epoch-{e}.pt")
        

    

if __name__ == "__main__":
    MIN_GPUS = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer',type=str, default="")
    args = parser.parse_args()

   
    train(0, 0, args.transfer)    
