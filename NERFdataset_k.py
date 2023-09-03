
from torch.utils.data import Dataset
import os
import pickle
import torch
from PIL import Image
import numpy as np
import torch
import random
import scipy.linalg as la


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class dataset(Dataset):
    
    def __init__(self, split, path='data/cars_train/', picklefile='data/cars.pickle', imgsize=128, nerf_view=None, normalize_first_view=True, nimg=4, seed=1, first_view=None, idx_offset=0):
        self.imgsize = imgsize
        self.path = path
        super().__init__()
        self.picklefile = pickle.load(open(picklefile, 'rb'))
        
        allthevid = sorted(list(self.picklefile.keys()))

        print(len(allthevid))
        
        random.seed(seed)
        random.shuffle(allthevid)
        if split == 'train':
            self.ids = allthevid[:int(len(allthevid)*0.9)]
        else:
            self.ids = allthevid[int(len(allthevid)*0.9):]
            
        self.nerf_view = nerf_view
        self.nimg=nimg
        self.normalize_first_view = normalize_first_view
        self.first_view=first_view

        self.idx_offset = idx_offset
        
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):

      
        if self.nerf_view:
            idx, first = self.nerf_view

        item = self.ids[(idx+self.idx_offset)%len(self.ids)]

        
        intrinsics_filename = os.path.join(self.path, item, 'intrinsics', self.picklefile[item][0][:-4] + ".txt")
        K = np.array(open(intrinsics_filename).read().strip().split()).astype(float).reshape((3,3))

        if self.nerf_view:          
            indices = [self.picklefile[item][first]] + random.sample(self.picklefile[item], k=self.nimg-1) #random.sample(self.picklefile[item], k=2)
        elif self.nimg is None:
            indices = self.picklefile[item]
            random.shuffle(indices)
        else:
            if self.first_view is None:
                indices = random.sample(self.picklefile[item], k=self.nimg)
            else:
                indices = [self.picklefile[item][self.first_view]] + random.sample(self.picklefile[item], k=self.nimg-1)
                
        imgs = []
        poses = []
        for i in indices:
            img_filename = os.path.join(self.path, item, 'rgb', i)
            img = Image.open(img_filename)
            if self.imgsize != 128:
                img = img.resize((self.imgsize, self.imgsize))
            img = np.array(img) / 255 * 2 - 1
            
            img = img.transpose(2,0,1)[:3].astype(np.float32)
            imgs.append(img)
            
            
            pose_filename = os.path.join(self.path, item, 'pose', i[:-4]+".txt")
            pose = np.array(open(pose_filename).read().strip().split()).astype(float).reshape((4,4))
            R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            pose[:3,:3] =  pose[:3,:3] @ R
            poses.append(pose)

        new_poses = []
        R0 = poses[0][:3,:3]
        T0 = poses[0][:3,3]
        d0 = la.norm(T0)

        camera_d = d0

        ref_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,d0],[0,0,0,1]])
        
        ref_transform = np.eye(4)
        ref_transform[:3,:3] = R0.T
        ref_transform[:3,3] = -R0.T @ T0 + np.array([0,0,d0])
        ref_transform = ref_transform

        if self.normalize_first_view:
            for p in poses:
                new_poses.append(ref_transform @ p)
        else:
            new_poses = poses

        imgs = np.stack(imgs, 0)
        poses = np.stack(new_poses, 0).astype(np.float32)
        R = poses[:, :3, :3]
        T = poses[:, :3, 3]
        
        intrinsics = np.array([K[0,0]/8, K[1,1]/8, K[0,2]/8, K[1,2]/8]).astype(np.float32)

        camera_k = K[0,0]/K[0,2]

        
        return {'imgs':imgs, 'poses':poses, 'intrinsics':intrinsics, 'K':K, 'camera_k': camera_k, 'camera_d': camera_d }


    
if __name__ == "__main__":
    
    d = dataset('train')
    dd = d[0]
    
    for ddd in dd.items():
        print(ddd[1].shape)
    print(dd['K'])
    print(dd['poses'])
