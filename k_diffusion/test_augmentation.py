

# Test augmentation module?

import matplotlib.pyplot as plt
import numpy as np
import torch

from imageio import imread

from augmentation import *

aug = KarrasAugmentationPipeline( a_prob=0.5)

aug_diff =KarrasDiffAugmentationPipeline()



image = imread('test.png')

mat, cond = aug.get_mat_cond(image)

print(mat, cond)

img1, img1_orig, _ = aug.apply(image, mat, cond)
img1b, img1_orig, _ = aug.apply_new(image, mat, cond)

img2, img2_orig, _ = aug_diff.apply_numpy(image, mat, cond)


plt.figure()
plt.imshow(image)
plt.figure()
plt.imshow(0.5*(img1.cpu().numpy()+1).transpose(1,2,0))

plt.figure()
plt.imshow(0.5*(img2.cpu().numpy()+1).transpose(1,2,0))

plt.figure()
plt.imshow(0.5*(img1b.cpu().numpy()+1).transpose(1,2,0))

plt.figure()
plt.imshow((img1-img2).cpu().numpy().transpose(1,2,0))

plt.figure()
plt.imshow((img1b-img2).cpu().numpy().transpose(1,2,0))

plt.show()