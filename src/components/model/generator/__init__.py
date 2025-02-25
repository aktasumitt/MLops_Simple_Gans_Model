import torch.nn as nn
import torch
from src.exception.exception import ExceptionNetwork, sys


class Generator(nn.Module):

    def __init__(self, channel_size, noise_size):
        super(Generator, self).__init__()

        

        self.gen_conv_layer = nn.Sequential(nn.ConvTranspose2d(noise_size, 512, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(),

                                            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),

                                            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(),
                                            
                                            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),

                                            nn.ConvTranspose2d(64, channel_size, kernel_size=4,stride=2, padding=1),  # (3,32,32)
                                            nn.Tanh())

    def forward(self, data):

        try:

            out = self.gen_conv_layer([data])
            
            return out
        except Exception as e:
            raise ExceptionNetwork(e, sys)
