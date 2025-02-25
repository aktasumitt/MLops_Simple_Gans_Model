import torch
from torchvision.utils import make_grid
from src.exception.exception import ExceptionNetwork,sys
import torchvision.utils as vutils
import numpy as np
import PIL.Image



def validation_model(model_generator,noise_size,device):  # Generator
        # We will see generated fake images for each labels 5 times
       
        try: 
                model_generator.eval()
                
                random_noise=torch.randn((20,noise_size,1,1)).to(device) 
                
                with torch.no_grad():
                        fake_img=model_generator(random_noise)
                
                # Tensorları grid haline getir (nrow belirterek sütun sayısını ayarlıyoruz)
                grid_tensor = vutils.make_grid(fake_img, nrow=5, normalize=True, scale_each=True)

                # Grid görüntüsünü NumPy formatına çevir
                grid_np = grid_tensor.cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                grid_np = (grid_np * 255).astype(np.uint8)  # Normalizasyonu kaldır, uint8 formatına çevir

                # NumPy array’i PIL Image'e çevir
                grid_pil = PIL.Image.fromarray(grid_np)

 
                return grid_pil
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)