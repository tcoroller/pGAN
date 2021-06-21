# Global packages
import os
import torch
import numpy as np
import cv2
import random
from torch.nn.functional import interpolate
from typing import Tuple, Dict
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

class GanMorphing():

    def __init__(self, netG, seed: int = None):
        self.netG = netG
        self.regions = {'c': 0, 't': 1, 'l': 2}
        self.seed = seed

    # =========================================================================
    #                   >>> Core functions <<<
    # =========================================================================

    def gen_image(self, z: torch.Tensor, size: tuple = (64, 64)) -> torch.Tensor:
        ''' Generate image from given latent variable and postprocess it '''
        return interpolate(self.netG(z), size=size, mode='bicubic').detach()

    def get_z(self, source: str = 'c') -> torch.Tensor:
        source = source.lower()
        assert source in self.regions, "Error: target not in ['c', 't', 'l']"
        return self.netG.noise_gen(1, region_fill=self.regions[source])[-1].detach()
    
    # =========================================================================
    #                   >>> Get synthetic - FIGURE 1 <<<
    # =========================================================================

    def get_fakes(self, n: int = 1) -> torch.Tensor:

        if self.seed is not None:
            torch.manual_seed(self.seed)
        tmp = []
        for key in self.regions:
            for _ in range(n):
                z = self.get_z(source=key)
                tmp.append(self.gen_image(z).squeeze(0))

        return torch.cat(tmp, axis=0).unsqueeze(1)  # batch*slize, 1, H, W
    
    # =========================================================================
    #                   >>> Morph C -> T -> L - FIGURE 2 <<<
    # =========================================================================

    def morphing(self, steps: int = 3):

        if self.seed is not None:
            torch.manual_seed(self.seed)

        # 1st morphing: cervical -> thoracic
        stacks_1, z_1 = self.interpolate(z=self.get_z(source='c'),
                                         target='t',
                                         steps=steps)
        # 2nd morphing: thoracic -> lumbar
        stacks_2, z_2 = self.interpolate(z=z_1[-1],
                                         target='l',
                                         steps=steps)

        # Concatenates stacks
        stacks = torch.cat([stacks_1[:-9, :, :], stacks_2], axis=0)
        # flip order, to start from cervival to lumbar
        stacks = torch.flip(stacks, dims=[0, 1, 2])

        # Concatenate the images tensors, adding grayscale channel information such as Batch*Slide, 1, H, W
        return stacks.unsqueeze(1)

    def interpolate(self, z: torch.Tensor, target: str = None, z2: torch.Tensor = None, steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:

        if z2 is None:
            assert target is not None, 'Error: please define target for c_interpolation'
            z_ = self.c_interpolation(z=z, steps=steps, target=target)
        else:
            assert z2 is not None, 'Error: please define z2 for z_interpolation'
            z_ = GanMorphing.z_interpolation(z1=z, z2=z2, steps=steps)

        images = [self.gen_image(x).squeeze(0) for x in z_]

        return torch.cat(images, axis=0), z_

    def c_interpolation(self, z: torch.Tensor, target: str = 'c', steps: int = 5) -> torch.Tensor:
        ''' Interpolate the conditional class only '''
        z_list = []

        # Checks
        source = GanMorphing.get_z_target(z)
        target = target.lower()
        assert target in self.regions, "Error: target not in ['c', 't', 'l']"
        assert source != target, 'Error, trying to interpolate for the same class'

        for i in np.linspace(0, 1, steps):
            z_ = z.clone().detach()
            z_[0, self.regions[source]] = (1-i)
            z_[0, self.regions[target]] = i
            z_list.append(z_)

        return z_list

    @staticmethod
    def z_interpolation(z1: torch.Tensor, z2: torch.Tensor, steps: int = 5) -> torch.Tensor:
        ''' Interpolate between 2 latent variable, Quadratic interpolation '''
        return [torch.lerp(z1, z2, i) for i in np.linspace(0, 1, steps)]

    @staticmethod
    def get_z_target(z):

        if z[0, 0].numpy() == 1:
            source = 'c'
        elif z[0, 1].numpy() == 1:
            source = 't'
        else:
            source = 'l'

        return source