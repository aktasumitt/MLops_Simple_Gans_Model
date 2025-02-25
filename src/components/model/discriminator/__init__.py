import torch.nn as nn
import torch
from src.exception.exception import ExceptionNetwork,sys


class Discriminator(nn.Module):
    def __init__(self,channel_size):
        super(Discriminator,self).__init__()
        
        
        
        self.disc_layer=nn.Sequential(nn.Conv2d(channel_size,64,kernel_size=3,stride=2,padding=1), # [(batch,3,32,32) + (batch,1,32,32)]--->batch,128,32,32
                                      nn.InstanceNorm2d(64,affine=True),
                                      nn.LeakyReLU(0.2), 
                                      
                                      nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
                                      nn.InstanceNorm2d(128,affine=True),
                                      nn.LeakyReLU(0.2),
                                      
                                      nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
                                      nn.InstanceNorm2d(256,affine=True),
                                      nn.LeakyReLU(0.2),    
                                      
                                      nn.Conv2d(256,1,kernel_size=3)) # batch,1,1,1
        
        
    def forward(self,data):
        try:            

            out=self.disc_layer(data)  
            return out
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
        