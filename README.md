# genvs_unofficial

Unofficial (partial) reimplementation of [GeNVS](https://github.com/NVlabs/genvs) by Chan et al.

## Introduction

After the success of text to image models, making 3D worlds and
environments from textual and image descriptions is one of the main
challenges for generative models.

Optimization-based models, such as DreamFusion, have shown great
promise at generating individual objects, but are computationally
expensive.  Other models have been developed, such as 3D-GANs, and
3D-diffusion models, which with enough training data should prove a
great way to generate individual objects. However, generating a whole
environment with multiple different objects is a more complex task.

Recently, [GeNVS](https://github.com/NVlabs/genvs) has been proposed
by Chan et al. This is a model which converts an image to a 3D
representation.  A denosing diffusion model, conditioned on the
rendering of this 3D representation from a novel camera position, is
used to generate a plausible image from this view. The diffusion model
allows the approach to refine missing details and generate unseen
details in this new view.

Whilst this model is here trained on single images, I believe that
this class of approach is the most promising route to generating 3D
environments.  This will later require substantial improvement in the
way in which multiple volumetric representations are combined.

The point of this repository is to make a simple attempt to reproduce
the model of Chan et al, using relatively limited computational
resources.

## NeRFs

Neural rendering fields - NeRFs for short - are a volumetric
representation of a 3D object or environment. Whilst they are commonly
implemented using a Deep Learning toolkit, they are not really a "Deep
Learning" method, as the Neural part refers to the representation of
the volumetric field, which at least for the initial implementations
was an implicit multilayer perceptron (MLP). These representations are
then optimized using similar algorithms to those used in Deep Learning
(Adam etc). More recent NeRF methods use voxels, triplanes or hash-tables
as a (partially) explicit representation of the volumetric field.

This volumetric field is used to generate images from specified camera
viewpoints using a volume rendering approach. At each point along
a camera ray, the volumetric field supplies a density and a emitted colour.
These are integrated along the ray (using an rendering equation) to generate
the colour of the pixel.

## The task (Novel View Synthesis)

Given a view of an object from one direction, Novel View Synthesis is
the task of reconstructing views of the object from other camera
positions.  This requires a model that is able to synthesise the
unseen parts of the object. As in many cases this problem is
underdetermined, with a distribution of different possible outputs, a
generative model which takes noise as an input is preferable to a
regression model. The objective for a regression model (typically
least squares or L1) optimizes the model to produce the mean or median
potential output, which can be grey and/or amorphous. Generative
models, which generally have a noise input, are able to generate a
distribution of outputs for a given conditioning input.

## Approach

GeNVS by Chan et al https://nvlabs.github.io/genvs/ combines three
main learnable components:

1. A network which projects images to features of a NeRF (aligned with
the camera frustrum). This NeRF is a voxel grid (like DVNO), of
dimension 64x64x32x16 (height x width x depth x channels).

2. A NeRF, used for volume rendering. This has two small MLPs which
map the interpolated features at each point in 3D to opacity
(1-channel) and a 16-channel latent feature. Using the standard NeRF
volume rendering approach, these volumetric features are used to
render a 16-channel, 64x64 image from the novel viewpoint.

3. A Denoising network, based on EDM of Karras et
al. https://github.com/NVlabs/edm . This uses the 16-channel rendered
image (bilinearly upscaled to 128x128) to condition a denoising
diffusion model, which generates the novel view. This is trained with
a denoising objective,

Part of the motivation for trying to reproduce this method, is that
the individual components are relatively lightweight (<100M
parameters), and at least in principle could be pre-trained
individually, before being combined and fine-tuned.

## Other approaches for this task

- PixelNerf https://arxiv.org/abs/2012.02190 . This is very similar to
  the network used in part 1, but generates a RGB NeRF which is used
  directly as the output. As this is a regression method, difficult to
  avoid blurriness / gray in the unseen portions of the model (as the
  objective tries to ensure that the model predicts the mean/median of
  the possible distribution).

- VisionNeRF https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/ . I
  believe that this essentially behaves like a very good PixelNeRF,
  with more powerful architecture. Still struggles with detail on the
  unobserved parts of the object (but copies colour across etc.)

- NerfDiff https://jiataogu.me/nerfdiff/ . This is a similar approach
  to GeNVS but uses an RGB triplane NeRF - camera but not frustrum
  aligned. This model uses a larger denoising UNet (400M-1B), with the
  option of cross attention to the input view. As this outputs RGB
  images it's possible to use a neat method a bit like
  score-distillation sampling during inference of the NeRF.

- 3DiM https://3d-diffusion.github.io/ .  UNet with cross attention
  between views. Pose encoding of camera ray start position and
  direction also supplied to the denosing model. This proves difficult
  to train.

- Zero123(XL) https://github.com/cvlab-columbia/zero123 . This is like
  3DiM but based on Stable Diffusion, conditioning generation on both
  the original view, an encoding of the original view, and the
  relative camera angle. This seeems to work pretty well, but being
  Stable-Diffusion based it is quite difficult to train. There is no
  explicit 3D representation of the object, so consistency is not
  guaranteed - see One-2-3-45 https://github.com/One-2-3-45/One-2-3-45
  for a neat method that quickly resolves this inconsistency.

## Differences with the paper

There are some elements that are either unclear in the paper,
deliberately changed, or just different by accident.


- View-conditioned denoiser uses the k-diffusion implementation of EDM
  at https://github.com/crowsonkb/k-diffusion . This is similar to
  that of EDM.

- Auxillary losses applied on volume rendered RGB images at novel
  views (first three channels of rendered latents).  This probably
  impedes optimal performance later, but permits pre-training of just
  the input network and NeRF, independent of the denoising
  network. This RGB rendering is by necessity blurry (and cannot deal
  with ambiguity), but seems to train faster than the diffusion model.

- Denoising diffusion model separately pre-trained (on the same
  training dataset as used for the whole model), and then combined
  with the input image->NeRF model.

- Also (small) auxillary losses applied to the occupancy of the
  rendered views. This attempted to stop the NeRF filling the entire
  volume of the frustrum, but at the weighting I used I believe it
  had little effect in practice.

- Image to NeRF model tends towards partial billboarding, with detail
  placed between the object and the camera. Attempted to correct this
  by additional loss penalizing differences in depth and opacity
  between the image where the source view is in the same position as
  the camera, and the image where the source view is in a different
  position. This didn't seem to help massively - the model just seemed
  to generate more background density. Training with a depth-objective
  would perhaps be a better approach, but the SRN dataset does not
  have depth images.

- Only one or two (not three) views supplied for each batch, in order
  to train at batch size>1 on consumer hardware. (For later training
  up to three views supplied.)

- Increased noise level in diffusion model - seems to help the model
  whilst training give better predictions of x0, conditioned on the
  NeRF renderings, far more quickly than the default setting - but may
  hinder sampling high resolution details later.

- Stochastic sampling - for whatever reason (insufficient training?
  discrepency between training and sampling noise levels?) the
  deterministic samplers perform poorly on this dataset. The model
  here uses 250 steps of the Euler-like method from the EDM paper.

- Simplistic autoregressive sampling - conditioning on supplied image,
  up to 4 intermediate images, and the previously generated
  image. Greatly improves sampling output but still flickers a bit
  with current trained model. Note that (to work well) sampling should
  start from a camera position near the supplied image and move
  gradually away from it,

## Training:

```
python train.py transfer=path_to_ckpt
```
Config file in config/config.yaml


## Data Preparation

Visit [SRN
repository](https://github.com/vsitzmann/scene-representation-networks),
download`cars_train.zip` and extract the downloaded files in
`/data/`. Here we use 90% of the training data for training and 10% as
the validation set.

From https://github.com/a6o/3d-diffusion-pytorch , there is a pickle
file that contains available view-png files per object.

Note that ShapeNet is only licensed for non-commercial use.

## Pre-trained Model Weights

Model weights available at https://huggingface.co/JFoz/genvs-unofficial

Training procedure somewhat complex - original model generated by
pretraining 64x64 diffusion model and image to NeRF models, combining
them and then further finetuning at resolutions 64 and 128.

Increased SNR levels (lonormal distribution with mean 1.0, standard
deviation 1.4) were used for this (see
k-configs/config_64_cars_noisy.json), and then subsequently dropped
later in training. This is much higher noise levels than in the
original paper, but we want good predictions of the denoised image at
high noise levels.

Noise levels dropped slightly (mean 0.5, standard deviation 1.4) for
further fine-tuning. Still considerably higher than those in the EDM
paper (mean -1.2 standard deviation 1.2). Scope for further
experimentation - this task is different to standard diffusion as the
conditioning with an extra image is far more informative than a text
prompt.

## Current results

```
python sample_views.py --transfer .
```

These clearly need a bit more training!

#### Conditioned on a single view

Conditioning image
![cars-conditioning-1-000000](https://github.com/jfozard/nvs_test/assets/4390954/0574042b-e372-4743-9433-d0cf209cd5a7)

Novel views generated (upper - denoised samples, lower- RGB renderings from NeRF)


Stochastic sampling

https://github.com/jfozard/nvs_test/assets/4390954/cd744bbd-9bdb-427a-a46c-be70f1a65e19

Deterministic sampling

https://github.com/jfozard/nvs_test/assets/4390954/fc45bd4d-0e34-455f-8d34-56ea20176c6b

Sampling progress (stochastic)

https://github.com/jfozard/nvs_test/assets/4390954/bdd2b8a1-8e31-4ee7-9949-6ca0efc9b98f

Sampling progress (deterministic)

https://github.com/jfozard/nvs_test/assets/4390954/c1a5acfd-2d34-4d55-8d14-95d71141a806

![cars-conditioning-1-000001](https://github.com/jfozard/nvs_test/assets/4390954/6882b749-dd47-486e-9377-8bb4c19da051)

https://github.com/jfozard/nvs_test/assets/4390954/a76015f1-2dc5-421c-8d17-405758c2cecd

https://github.com/jfozard/nvs_test/assets/4390954/e2c8f175-c4a3-465a-ae14-4a39a11ff506

https://github.com/jfozard/nvs_test/assets/4390954/5e8cd52f-aff9-4b36-8074-bdbee2d35eb6

https://github.com/jfozard/nvs_test/assets/4390954/8c0aecd0-1744-4a0c-bfab-8803265905b4


#### Unconditional samples (Supply pure noise conditioning image to diffusing model)

Stochastic

![uc-step-1-000000-250](https://github.com/jfozard/nvs_test/assets/4390954/590c4fd7-9b47-4390-bdfb-ef50446639e0)

https://github.com/jfozard/nvs_test/assets/4390954/6fedfdae-7ae3-4e95-bb00-6af90235b3e0

Deterministic

![uc_det-step-1-000000-250](https://github.com/jfozard/nvs_test/assets/4390954/6771b11d-23ff-48f4-8928-993900186c5e)

https://github.com/jfozard/nvs_test/assets/4390954/2aa18fa4-3317-4d29-b3e8-97bf3f337f4e

### Autoregressive sampling

This produces much better results than sampling from a single
view. Strongly suggests this is required for decent levels of
multi-view consistency. Still struggles a little bit with flickering,
and consistency of details between the different sides of the
vehicle. Unclear if this is due to insufficient training of the Image
-> NeRF network, or a deficiency because pairs of features on opposite
sides of the vehicle can never appear together in a single image. In
the latter case, cross attention between views (3DiM, nerfdiff) may be
a sensible addition to the denoising model.

First frame of each video is the conditioning image.

https://github.com/jfozard/nvs_test/assets/4390954/716969c4-8061-45c0-95dd-47ade62fb305

https://github.com/jfozard/nvs_test/assets/4390954/e9bae498-50ff-49f2-98e4-43930158a446

https://github.com/jfozard/nvs_test/assets/4390954/580a5ba1-8354-4f5d-a674-f4f08ca2a208

https://github.com/jfozard/nvs_test/assets/4390954/a28173b0-9469-449c-a1a2-3499038e5629

https://github.com/jfozard/nvs_test/assets/4390954/82b1a47d-571d-4976-aa2c-0892dbca2c13



## TODO

- Further training of model on a real multi-GPU system.

- Investigate inference strategy further - which images to retain in
  conditioning, and whether to resample views?

- Increase augmentation amount - current denoising model struggles
  with views which differ substantially from training set.

- Train on larger, more general dataset.

- Explore noise range schedules during training - start with fairly
  high noise levels and drop over time.

- Also explore LR schedule.

- Get a decent pixelNeRF to use as a starting point for training

- Similarly, obtain a decent k-diffusion model to fine-tune rather than train from
  scratch.

## Acknowledgements

K-diffusion from Katherine Crawson and others https://github.com/crowsonkb/k-diffusion

NeRF rendering using an old version of ashawkey's excellent https://github.com/ashawkey/stable-dreamfusion

Some data pipeline from https://github.com/a6o/3d-diffusion-pytorch and https://github.com/halixness/distributed-3d-diffusion-pytorch





