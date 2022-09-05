"""
DynamicsAE: A deep learning-based framework to uniquely identify an uncorrelated,
isometric and meaningful latent representation. Code maintained by Dedi.

Read and cite the following when using this method:
https://arxiv.org/abs/2209.00905
"""
import torch
from torch import nn
import numpy as np

class fc_encoder(nn.Module):

    def __init__(self, z_dim, data_shape, device, neuron_num1=16):

        super(fc_encoder, self).__init__()

        self.z_dim = z_dim

        self.neuron_num1 = neuron_num1

        self.data_shape = data_shape

        self.eps = 1e-10
        self.device = device

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(np.prod(self.data_shape), self.neuron_num1),
            nn.ReLU())

        self.encoder = self._encoder_init()

        self.encoder_output = nn.Linear(self.neuron_num1, self.z_dim)

    def _encoder_init(self):
        modules = []
        for _ in range(2):
            modules += [nn.Linear(self.neuron_num1, self.neuron_num1)]
            modules += [nn.ReLU()]

        return nn.Sequential(*modules)

class conv_encoder(nn.Module):

    def __init__(self, z_dim, data_shape, device, neuron_num1=16):

        super(conv_encoder, self).__init__()

        self.z_dim = z_dim

        self.neuron_num1 = neuron_num1

        self.data_shape = data_shape

        self.eps = 1e-10
        self.device = device

        self.encoder_input_layer = nn.Sequential(
            nn.Conv2d(data_shape[0], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True))

        self.encoder = self._encoder_init()

        self.encoder_output = nn.Linear(self.neuron_num1, self.z_dim)

    def _encoder_init(self):

        modules = []
        modules += [nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)]
        modules += [nn.ReLU(True)]
        modules += [nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)]
        modules += [nn.ReLU(True)]
        modules += [nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)]
        modules += [nn.ReLU(True)]
        modules += [nn.Conv2d(64, self.neuron_num1//4//4, kernel_size=1)]
        modules += [nn.ReLU()]

        return nn.Sequential(*modules)

class fc_decoder(nn.Module):

    def __init__(self, z_dim, output_shape, data_shape, neuron_num2=16):

        super(fc_decoder, self).__init__()

        self.z_dim = z_dim
        self.output_shape = output_shape

        self.neuron_num2 = neuron_num2

        self.data_shape = data_shape

        self.decoder_input_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.neuron_num2),
            nn.ReLU())

        self.decoder_output = nn.Linear(self.neuron_num2, self.output_shape[0])

        self.decoder = self._decoder_init()

    def _decoder_init(self):
        modules = []
        for _ in range(2):
            modules += [nn.Linear(self.neuron_num2, self.neuron_num2)]
            modules += [nn.ReLU()]

        return nn.Sequential(*modules)

class deconv_decoder(nn.Module):

    def __init__(self, z_dim, output_shape, data_shape, neuron_num2=16):

        super(deconv_decoder, self).__init__()

        self.z_dim = z_dim
        self.output_shape = output_shape

        self.neuron_num2 = neuron_num2

        self.data_shape = data_shape

        self.decoder_input_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.neuron_num2),
            nn.ReLU(),
            nn.Linear(self.neuron_num2, 1024),
            nn.ReLU())

        self.decoder_output = nn.Sequential(
            nn.ConvTranspose2d(32, self.output_shape[0], kernel_size=4, stride=2, padding=1))

        self.decoder = self._decoder_init()

    def _decoder_init(self):
        # cross-entropy MLP decoder
        # output the probability of future state
        modules = []
        modules += [nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)]
        modules += [nn.ReLU(True)]
        modules += [nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)]
        modules += [nn.ReLU(True)]
        modules += [nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)]
        modules += [nn.ReLU(True)]

        return nn.Sequential(*modules)

class Langevin_prior(nn.Module):

    def __init__(self, z_dim, data_shape, device, neuron_num=64):

        super(Langevin_prior, self).__init__()

        self.z_dim = z_dim

        self.data_shape = data_shape

        self.eps = 1e-10
        self.device = device

        # learning diffusion
        self.prior_logdiff_net = nn.Sequential(
            nn.Linear(self.z_dim, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, self.z_dim))

        # learning force
        self.prior_force_net = nn.Sequential(
            nn.Linear(self.z_dim, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, self.z_dim))

    def prior_logdiff(self, z):
        logdiff = self.prior_logdiff_net(z)
        return logdiff

    def prior_force(self, z):
        force = self.prior_force_net(z)
        return force