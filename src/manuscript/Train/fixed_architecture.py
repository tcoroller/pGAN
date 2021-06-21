# pylint: disable=W0223, E1101  # torch: Method '_forward_unimplemented' is abstract in class 'Module' but is not overridden
'''
Basic GAN architecture used to generate synthetic data.
The architecture is a AC-GAN where:
    - The Generator is a CNN Decoder with 4x4 kernels, batchnorm and ReLU activation
    - The Discriminator is a CNN Encoder with 4x4 kernels, batchnorm and leaky ReLU activation
    (slope 0.2)
The input Z vector for generator can be sampled within the class with the function noise_gen()
The label distribution prior can be added directly (with set_probs()) or within as
an argument for the noise_gen function.
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

# Default intialization for Conv layers and Batchnorm
def default_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    '''Generator
    init():
        - Hyperparmater intilialization (feature dim for z vector, GPU or CPU and network size)
        - Network architecture (5 layer Decoder with 4x4 transpose convolutions, BatchNorm,
        Relu activations, output with Tanh)
    forward(): network inference
    load(): loading weights from pickle/torch file
    set_probs(): list of probability for each region: prob, in this case prob=[0.2, 0.55, 0.25]
    noise_gen(): generate z vector for generator input
    '''
    # pylint: disable=R0913  # Too many arguments
    def __init__(self, feature_dim, nc, num_classes_region=3, net_size=128,
                 batch_norm=False, weights_init=default_weights_init, probs=None, device=torch.device('cpu')):
        super(Generator, self).__init__()
        self.net_size = net_size
        self.feature_dim = feature_dim
        self.z_dim = self.feature_dim + num_classes_region  # feature_dim + region_dim(= 3)
        self.probs = probs  # probability distributions for regions (3-dim)
        self.device = device

        def tconv_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            return layers

        self.model = nn.Sequential(
            *tconv_block(self.z_dim, net_size*8, 4, 1, 0),
            *tconv_block(net_size*8, net_size*4, 4, 2, 1),
            *tconv_block(net_size*4, net_size*2, 4, 2, 1),
            *tconv_block(net_size*2, net_size, 4, 2, 1),
            nn.ConvTranspose2d(net_size, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, z):
        z = z.view(-1, self.z_dim, 1, 1)
        x = self.model(z)
        return x

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['netG'])

    def set_probs(self, probs):
        self.probs = probs

    def noise_gen(self, size, device=None, probs=None, region_fill=None):
        if device is None:
            device = self.device
        if probs is None:
            probs = self.probs

        cr = np.random.choice(3, size=size, p=probs['region']) if region_fill is None else np.ones(size) * region_fill
        cr = torch.from_numpy(cr).to(device).long()

        z = torch.cat([one_hot(cr, num_classes=3).float(),
                       torch.randn(size=(size, self.feature_dim), device=device)], dim=1)
        return cr, z

class Discriminator(nn.Module):
    '''Discriminator
    init():
        - Hyperparmater intilialization (feature dim for z vector, GPU or CPU and network size)
        - Network architecture (5 layer Encoder with 4x4 convolutions, BatchNorm,
        leaky (0.2) Relu activations, output with 2 fully connected layers with Sigmoid,
            - one fully connected for predicting adversarial (real or fake)
            - one fully connected for predicting class (region Cervical, Thoracic or Lumbar)
    forward(): network inference
    load(): loading weights from pickle/torch file
    '''
    def __init__(self, nc, num_classes_region=3, net_size=64, batch_norm=False, neg_slope=0.2,
                 weights_init=default_weights_init):
        super(Discriminator, self).__init__()
        self.net_size = net_size

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(neg_slope, True))
            return layers

        self.model = nn.Sequential(
            nn.Conv2d(nc, net_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),
            *conv_block(net_size, net_size*2, 4, 2, 1),
            *conv_block(net_size*2, net_size*4, 4, 2, 1),
            *conv_block(net_size*4, net_size*8, 4, 2, 1),
            nn.Conv2d(net_size*8, net_size, 4, 1, 0, bias=False)
        )

        # discriminator (adversarial loss)
        self.fc_d = nn.Sequential(
            nn.Linear(net_size, 1),
            nn.Sigmoid(),
        )

        # aux-classifier: region
        self.fc_cr = nn.Sequential(
            nn.Linear(net_size, num_classes_region),
            nn.Softmax(dim=1),
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.model(x).view(-1, self.net_size)
        d = self.fc_d(x).view(-1, 1).squeeze(1)
        cr = self.fc_cr(x)
        return d, cr

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['netD'])
