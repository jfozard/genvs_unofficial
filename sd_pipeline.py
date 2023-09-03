#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
#    StableDiffusionControlNetPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)



from pipeline_controlnet import StableDiffusionControlNetPipeline

from controlnet import ControlNetModel

#from diffusers.models.controlnet import  ControlNetConditioningEmbedding
from controlnet import  ControlNetConditioningEmbedding

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from torch import nn

import k_diffusion as K
from k_diffusion.external import CompVisDenoiser


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

class SDPipeline(nn.Module):

    def __init__(self, 
                 pretrained_model_name_or_path='segmind/tiny-sd',#runwayml/stable-diffusion-v1-5',
                 #pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
                 revision = None,
                 unlock_up_blocks=False,
                 enable_xformers_memory_efficient_attention = False,
                 gradient_checkpointing=False,
                 device='cuda',
                 cond_channels=16,
                 zero_uncond=False,
                 cfg=5.0,
    ):

        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention
        self.revision = revision

        self.snr_gamma = 10.0 #1.0(restart3)

        self.cfg = cfg

        self.zero_uncond = zero_uncond
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        
        #self.noise_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        #self.noise_scheduler = EulerDiscreteScheduler.from_config(self.noise_scheduler.config)
        
        self.text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
        )
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=revision
        )
        self.weight_dtype = torch.float32

        print('UNet', self.unet)

        self.controlnet = ControlNetModel.from_unet(self.unet, num_classes=2).to(device, dtype=self.weight_dtype)

        self.controlnet.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels = 320,
            conditioning_channels = cond_channels,
            block_out_channels=(320, 640, 1280, 1280),
        ).to(device, dtype=self.weight_dtype)

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        if unlock_up_blocks:
            for b in self.unet.up_blocks:
                b.requires_grad_(True)
                
        self.text_encoder.requires_grad_(False)
        self.controlnet.train()

        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                self.controlnet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        else:
            self.unet.disable_xformers_memory_efficient_attention()
            self.controlnet.disable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()


        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(device, dtype=self.weight_dtype)
        self.unet.to(device, dtype=self.weight_dtype)
        self.text_encoder.to(device, dtype=self.weight_dtype)

    def parameters(self):
        return self.controlnet.parameters()
    
    def tokenize_captions(self, captions, is_train=True):
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def train_step(self, pixel_values, conditioning_pixel_values, cond_flag=0, aug_cond=None, offset=0.1):
        bsz = pixel_values.shape[0]
        input_ids = self.tokenize_captions(['a car']*bsz).cuda()

        cond_flag = cond_flag.to(pixel_values.device)

        if self.zero_uncond:
            conditioning_pixel_values = torch.where(cond_flag[:,None,None,None], torch.zeros_like(conditioning_pixel_values), conditioning_pixel_values)
        else:
            conditioning_pixel_values = torch.where(cond_flag[:,None,None,None], torch.rand_like(conditioning_pixel_values), conditioning_pixel_values)


        
            
        latents = self.vae.encode(pixel_values.to(dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)+0.1*torch.randn((latents.shape[0],4,1,1), dtype=latents.dtype, device=latents.device)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids)[0]


        
        controlnet_image = conditioning_pixel_values.to(dtype=self.weight_dtype)

        """
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )
        """

        class_labels = cond_flag.to(torch.int32) #torch.tensor((bsz,), device=latents.device, dtype=torch.int32).fill_(1 if cond_flag else 0)

        down_block_res_samples, _ = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )
        
        #print('down', [d.shape for d in down_block_res_samples])
        #down_block_res_samples = down_block_res_samples[1::2]
        #print('down', [d.shape for d in down_block_res_samples])
        
        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[
                sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
            ],
            #mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
        ).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        
        snr = self.compute_snr(timesteps)
        mse_loss_weights = (
            torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        # We first calculate the original loss. Then we mean over the non-batch dimensions and
        # rebalance the sample-wise losses with their respective loss weights.
        # Finally, we take the mean of the rebalanced loss.
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()        
        #loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    @torch.no_grad()
    def sample(self, conditioning_pixel_values, sampling_timesteps=50, stochastic=True, unconditional=False, cfg=5.0, sampler_name="ddpm"):

        n = conditioning_pixel_values.shape[0]

        if unconditional:
            if self.zero_uncond:
                conditioning_pixel_values = torch.zeros_like(conditioning_pixel_values)
            else:
                conditioning_pixel_values = torch.rand_like(conditioning_pixel_values)

        input_prompts = ['a car']*n
        
        controlnet = self.controlnet

        all_samplers = { 'ddpm':DDPMScheduler,
              'euler_a': EulerAncestralDiscreteScheduler,
              'euler': EulerDiscreteScheduler,
              'unipc': UniPCMultistepScheduler }

        
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=self.revision,
            torch_dtype=self.weight_dtype,
            local_files_only=True,
        )
        #pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        #pipeline.scheduler = self.noise_scheduler

        print('sampler', sampler_name)
        pipeline.scheduler = all_samplers[sampler_name].from_config(self.noise_scheduler.config)

        print('config', pipeline.scheduler.config)
        
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        if self.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        generator = None

        validation_images = conditioning_pixel_values
        #validation_prompts = [""] * n


        with torch.autocast("cuda"):
            images= pipeline(
                input_prompts, validation_images, num_inference_steps=sampling_timesteps, generator=generator, guidance_scale=cfg, class_labels=torch.tensor((n,), device=self.device, dtype=torch.int32).fill_(1 if unconditional else 0)
                    ).images
        return images        



    @torch.no_grad()
    def sample_all_orig(self, conditioning_pixel_values, sampling_timesteps=50, stochastic=True, unconditional=False, sampler_name='ddpm', cfg=1):

        all_samplers = { 'ddpm':DDPMScheduler,
              'euler_a': EulerAncestralDiscreteScheduler,
              'euler': EulerDiscreteScheduler,
              'unipc': UniPCMultistepScheduler }

        n = conditioning_pixel_values.shape[0]

        if unconditional:
            if self.zero_uncond:
                conditioning_pixel_values = torch.zeros_like(conditioning_pixel_values)
            else:
                conditioning_pixel_values = torch.rand_like(conditioning_pixel_values)

                
        input_prompts = ['a car']*n
        
        controlnet = self.controlnet

        controlnet.eval()
        
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=self.revision,
            torch_dtype=self.weight_dtype,
            local_files_only=True,
        )

        
        #if sampler:
        #    pipeline.scheduler = samplers[sampler].from_config(self.noise_scheduler.config)
        #else:
        #    pipeline.scheduler = self.noise_scheduler

        print('sampler', sampler_name)
        pipeline.scheduler = all_samplers[sampler_name].from_config(self.noise_scheduler.config)
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        progress = []
        
        if self.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        generator = None

        validation_images = conditioning_pixel_values
        #validation_prompts = [""] * n

        def callback(step, timestep, latents):
            progress.append(latents) #self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])

        with torch.autocast("cuda"):
            latents = pipeline(
                input_prompts, validation_images, num_inference_steps=sampling_timesteps, generator=generator, guidance_scale=cfg, class_labels=torch.tensor((n,), device=self.device, dtype=torch.int32).fill_(1 if unconditional else 0), output_type="latent", callback=callback).images
            progress.append(latents)
        progress = [ self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] for latents in progress]
        #print([p.shape for p in progress])
        progress = torch.stack(progress, dim=0)
        #print(progress.shape)

        controlnet.train()
        
        return progress


    @torch.no_grad()
    def sample_all_multi_guidance(self, conditioning_pixel_values, sampling_timesteps=50, stochastic=True, unconditional=False, sampler_name='ddpm', cfg=1, cfg2=1, churn=0.0, latents=None, starting_timestep=None):

        controlnet_both_sides = False
        
        n = conditioning_pixel_values.shape[0]

        do_cfg = (cfg!=1) and (cfg!=1.0)

            
                
        input_prompts = ['a car']*n
        
        controlnet = self.controlnet

        controlnet.eval()

        noise_scheduler = self.noise_scheduler


        class_labels = torch.zeros((n,), device=self.device, dtype=torch.int32).fill_(1)


        
        if do_cfg:
            class_labels = torch.cat([torch.zeros_like(class_labels), class_labels], dim=0)
    

            
        if do_cfg:
            print('CFG')

            input_ids = self.tokenizer(
                input_prompts + input_prompts + [""]*n, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.cuda()
            encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
            input_ids = self.tokenizer(
                input_prompts + input_prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.cuda()
            controlnet_encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
        

        if do_cfg:
            if self.zero_uncond:
                unconditioning_pixel_values = torch.zeros_like(conditioning_pixel_values)
            else:
                unconditioning_pixel_values = torch.rand_like(conditioning_pixel_values)
            conditioning_pixel_values = torch.cat([conditioning_pixel_values, unconditioning_pixel_values], dim=0)

        controlnet_image = conditioning_pixel_values.to(dtype=self.weight_dtype)

        weight_dtype = self.weight_dtype

        class ControlNetDenoiser(K.external.DiscreteEpsDDPMDenoiser):
            """A wrapper for CompVis diffusion models."""

            def __init__(self, controlnet, unet, alphas_cumprod, cfg, quantize=False, device='cpu'):
                super().__init__(unet, alphas_cumprod, quantize=quantize)
                self.controlnet = controlnet
                self.cfg = cfg
                self.cfg2 = cfg2
                
            def get_eps(self, *args, **kwargs):


                #eps_all = self.inner_model(sample, t, encoder_hidden_states=kwargs['encoder_hidden_states']).sample

                if do_cfg:

                    sample = torch.cat([args[0]]*2, dim=0)
                    t = torch.cat([args[1]]*2, dim=0)


                    down_block_res_samples, _ = self.controlnet(
                        sample,
                        t,
                        encoder_hidden_states=controlnet_encoder_hidden_states,
                        class_labels=class_labels,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    down_block_res_samples = [torch.cat([d, torch.zeros_like(d[:d.shape[0]//2])]) for d in down_block_res_samples]
                    
                    sample = torch.cat([args[0]]*3, dim=0)
                    t = torch.cat([args[1]]*3, dim=0)

                    
                    # Predict the noise residual
                    eps_all = self.inner_model(
                        sample,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                    ).sample
                    eps, eps_no_img, eps_uc = eps_all.chunk(3)
    
                    return eps*self.cfg2 + eps_no_img*(1-self.cfg2 + self.cfg-1) - eps_uc*(self.cfg-1)
                    

                    
        model = ControlNetDenoiser(self.controlnet, self.unet, torch.tensor(noise_scheduler.alphas_cumprod.numpy()), cfg)
                        
        model = model.to(self.device)

        sigmas = model.get_sigmas(sampling_timesteps)
        print('sigmas', sigmas)
        random_sigmas =  torch.exp(-1.2 + 1.2*torch.randn((100,)))
        print('random sigmas', random_sigmas)

        progress = []
        if latents is not None:
            progress.append(latents)
        
        def callback(u):
            progress.append(u['denoised'])
        
        def sample_fn(n):
            if starting_timestep is None:
                x = torch.randn([n, 4, 16, 16], device=self.device) * sigmas[0]
                x_0 = K.sampling.sample_euler(model, x, sigmas, s_tmin=0., s_tmax=float('inf'), s_noise=1.000, s_churn=churn*(40/256)*sampling_timesteps, extra_args={'encoder_hidden_states': encoder_hidden_states}, callback=callback)
            else:
                s = sigmas[starting_timestep]
                x = (torch.randn([n, 4, 16, 16], device=self.device) * s + latents)#/torch.sqrt(1+s**2)
                x_0 = K.sampling.sample_euler(model, x, sigmas[starting_timestep:], s_tmin=0., s_tmax=float('inf'), s_noise=1.000, s_churn=churn*(40/256)*sampling_timesteps, extra_args={'encoder_hidden_states': encoder_hidden_states}, callback=callback)
                
            return x_0
            
        x_0 = sample_fn(n)
        progress.append(x_0)
        
        print(x_0.min(), x_0.max(), x_0.mean(), x_0.var())
            
        progress = [ self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] for latents in progress]
        #print([p.shape for p in progress])
        progress = torch.stack(progress, dim=0)
        #print(progress.shape)

        controlnet.train()
        
        return progress



    @torch.no_grad()
    def sample_all(self, conditioning_pixel_values, sampling_timesteps=50, stochastic=True, unconditional=False, sampler_name='ddpm', cfg=1, churn=0.0, latents=None, starting_timestep=None):

        controlnet_both_sides = False
        
        n = conditioning_pixel_values.shape[0]

        do_cfg = (cfg!=1) and (cfg!=1.0)

            
                
        input_prompts = ['a car']*n
        
        controlnet = self.controlnet

        controlnet.eval()

        noise_scheduler = self.noise_scheduler


        class_labels = torch.zeros((n,), device=self.device, dtype=torch.int32).fill_(1 if unconditional else 0)


        
        if do_cfg and controlnet_both_sides:
            class_labels = torch.cat([class_labels, torch.zeros_like(class_labels).fill_(1)], dim=0)
    

            
        if do_cfg:
            print('CFG')

            if controlnet_both_sides:
                input_ids = self.tokenizer(
                input_prompts + input_prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.cuda()
                encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
            else:
                input_ids = self.tokenizer(
                input_prompts + [""]*n, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.cuda()
                encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
                input_ids = self.tokenizer(
                input_prompts , max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.cuda()
                controlnet_encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
                
                

                

        else:
            input_ids = self.tokenizer(
                input_prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.cuda()
            
            encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
        

        if unconditional:
            if self.zero_uncond:
                conditioning_pixel_values = torch.zeros_like(conditioning_pixel_values)
            else:
                conditioning_pixel_values = torch.rand_like(conditioning_pixel_values)

        if do_cfg and controlnet_both_sides:
            if self.zero_uncond:
                unconditioning_pixel_values = torch.zeros_like(conditioning_pixel_values)
            else:
                unconditioning_pixel_values = torch.rand_like(conditioning_pixel_values)
            conditioning_pixel_values = torch.cat([conditioning_pixel_values, unconditioning_pixel_values], dim=0)

        controlnet_image = conditioning_pixel_values.to(dtype=self.weight_dtype)

        weight_dtype = self.weight_dtype
        class ControlNetDenoiser(K.external.DiscreteEpsDDPMDenoiser):
            """A wrapper for CompVis diffusion models."""

            def __init__(self, controlnet, unet, alphas_cumprod, cfg, quantize=False, device='cpu'):
                super().__init__(unet, alphas_cumprod, quantize=quantize)
                self.controlnet = controlnet
                self.cfg = cfg
                
            def get_eps(self, *args, **kwargs):


                #eps_all = self.inner_model(sample, t, encoder_hidden_states=kwargs['encoder_hidden_states']).sample

                if do_cfg and controlnet_both_sides:

                    sample = torch.cat([args[0]]*2, dim=0)
                    t = torch.cat([args[1]]*2, dim=0)

                    print(sample.shape, t.shape, encoder_hidden_states.shape, class_labels.shape, controlnet_image.shape)
                    down_block_res_samples, _ = self.controlnet(
                        sample,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        class_labels=class_labels,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                
                    # Predict the noise residual
                    eps_all = self.inner_model(
                        sample,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                    ).sample
                    eps, eps_uc = eps_all.chunk(2)
    
                    return eps*self.cfg - eps_uc*(self.cfg-1)
                    

                elif do_cfg:
                    down_block_res_samples, _ = self.controlnet(
                        args[0],
                        args[1],
                        encoder_hidden_states=controlnet_encoder_hidden_states,
                        #encoder_hidden_states=encoder_hidden_states,
                        class_labels=class_labels,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    down_block_res_samples = [torch.cat([d, torch.zeros_like(d)]) for d in down_block_res_samples]
                    sample = torch.cat([args[0]]*2, dim=0)
                    t = torch.cat([args[1]]*2, dim=0)

                
                    # Predict the noise residual
                    eps_all = self.inner_model(
                    sample,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
            #mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                    ).sample
                    eps, eps_uc = eps_all.chunk(2)
    
                    return eps*self.cfg - eps_uc*(self.cfg-1)
                else:
                    sample = args[0]
                    t = args[1]

                    down_block_res_samples, _ = self.controlnet(
                        sample,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        class_labels=class_labels,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )


                    eps = self.inner_model(
                    sample,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
            #mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                    ).sample
                    return eps
                    
        model = ControlNetDenoiser(self.controlnet, self.unet, torch.tensor(noise_scheduler.alphas_cumprod.numpy()), cfg)
                        
        model = model.to(self.device)

        sigmas = model.get_sigmas(sampling_timesteps)
        print('sigmas', sigmas)
        random_sigmas =  torch.exp(-1.2 + 1.2*torch.randn((100,)))
        print('random sigmas', random_sigmas)
        print('latents', latents.shape if latents is not None else latents)
            
        """
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )
        """

        """


        """
        progress = []
        if latents is not None:
            progress.append(latents)
        
        def callback(u):
            progress.append(u['denoised'])
        
        def sample_fn(n):
            if starting_timestep is None:
                x = torch.randn([n, 4, 16, 16], device=self.device) * sigmas[0]
                x_0 = K.sampling.sample_euler(model, x, sigmas, s_tmin=0., s_tmax=float('inf'), s_noise=1.000, s_churn=churn*(40/256)*sampling_timesteps, extra_args={'encoder_hidden_states': encoder_hidden_states}, callback=callback)
            else:
                s = sigmas[starting_timestep]
                x = (torch.randn([n, 4, 16, 16], device=self.device) * s + latents)#/torch.sqrt(1+s**2)
                x_0 = K.sampling.sample_euler(model, x, sigmas[starting_timestep:], s_tmin=0., s_tmax=float('inf'), s_noise=1.000, s_churn=churn*(40/256)*sampling_timesteps, extra_args={'encoder_hidden_states': encoder_hidden_states}, callback=callback)
                
            return x_0
            
        x_0 = sample_fn(n)
        progress.append(x_0)
        
        print(x_0.min(), x_0.max(), x_0.mean(), x_0.var())
            
        progress = [ self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] for latents in progress]
        #print([p.shape for p in progress])
        progress = torch.stack(progress, dim=0)
        #print(progress.shape)

        controlnet.train()
        
        return progress

    @torch.no_grad()
    def denoise_all(self, conditioning_pixel_values, sampling_timesteps=50, stochastic=True, unconditional=False, sampler_name='ddpm', cfg=1, churn=0.0, latents=None, starting_timestep=None):

        controlnet_both_sides = False
        
        n = conditioning_pixel_values.shape[0]

        do_cfg = (cfg!=1) and (cfg!=1.0)

            
                
        input_prompts = ['a car']*n
        
        controlnet = self.controlnet

        controlnet.eval()

        noise_scheduler = self.noise_scheduler


        class_labels = torch.zeros((n,), device=self.device, dtype=torch.int32).fill_(1)


        
        if do_cfg:
            class_labels = torch.cat([torch.zeros_like(class_labels), class_labels], dim=0)
    

            
        if do_cfg:
            print('CFG')

            input_ids = self.tokenizer(
                input_prompts + input_prompts + [""]*n, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.cuda()
            encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
            input_ids = self.tokenizer(
                input_prompts + input_prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.cuda()
            controlnet_encoder_hidden_states = self.text_encoder(input_ids)[0].cuda()
        

        if do_cfg:
            if self.zero_uncond:
                unconditioning_pixel_values = torch.zeros_like(conditioning_pixel_values)
            else:
                unconditioning_pixel_values = torch.rand_like(conditioning_pixel_values)
            conditioning_pixel_values = torch.cat([conditioning_pixel_values, unconditioning_pixel_values], dim=0)

        controlnet_image = conditioning_pixel_values.to(dtype=self.weight_dtype)

        weight_dtype = self.weight_dtype

        class ControlNetDenoiser(K.external.DiscreteEpsDDPMDenoiser):
            """A wrapper for CompVis diffusion models."""

            def __init__(self, controlnet, unet, alphas_cumprod, cfg, quantize=False, device='cpu'):
                super().__init__(unet, alphas_cumprod, quantize=quantize)
                self.controlnet = controlnet
                self.cfg = cfg
                
            def get_eps(self, *args, **kwargs):


                #eps_all = self.inner_model(sample, t, encoder_hidden_states=kwargs['encoder_hidden_states']).sample

                if do_cfg:

                    sample = torch.cat([args[0]]*2, dim=0)
                    t = torch.cat([args[1]]*2, dim=0)


                    down_block_res_samples, _ = self.controlnet(
                        sample,
                        t,
                        encoder_hidden_states=controlnet_encoder_hidden_states,
                        class_labels=class_labels,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    down_block_res_samples = [torch.cat([d, torch.zeros_like(d[:d.shape[0]//2])]) for d in down_block_res_samples]
                    
                    sample = torch.cat([args[0]]*3, dim=0)
                    t = torch.cat([args[1]]*3, dim=0)

                    
                    # Predict the noise residual
                    eps_all = self.inner_model(
                        sample,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                    ).sample
                    eps, eps_no_img, eps_uc = eps_all.chunk(3)
    
                    return eps #+ eps_no_img*(self.cfg-1) - eps_uc*(self.cfg-1)
                    

                    
        model = ControlNetDenoiser(self.controlnet, self.unet, torch.tensor(noise_scheduler.alphas_cumprod.numpy()), cfg)
                        
        model = model.to(self.device)

        sigmas = model.get_sigmas(sampling_timesteps)
        print('sigmas', sigmas)
        random_sigmas =  torch.exp(-1.2 + 1.2*torch.randn((100,)))
        print('random sigmas', random_sigmas)

            
        """
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )
        """

        """


        """
        progress = []
        if latents is not None:
            progress.append(latents)
        

        for t in range(0, 1000, 100):
            t = torch.tensor([t])
            snr = self.compute_snr(t)
            g = torch.sqrt(1/snr)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
            )
            print(t, snr, mse_loss_weights, g)


        for i, s in enumerate(sigmas):


            print(i,s)
            x = (torch.randn([n, 4, 16, 16], device=self.device) * s + latents)

            x_0 = model(x, torch.tensor([s]*n).cuda())

            progress.append(x_0)
            
        
        print(x_0.min(), x_0.max(), x_0.mean(), x_0.var())
            
        progress = [ self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] for latents in progress]
        #print([p.shape for p in progress])
        progress = torch.stack(progress, dim=0)
        #print(progress.shape)

        controlnet.train()
        
        return progress

