import torch
from torchvision.utils import make_grid
from src.exception.exception import ExceptionNetwork,sys
import torchvision.utils as vutils
import numpy as np
import PIL.Image
from src.utils import load_obj
from src.entity.config_entity import TestConfig

class TestModule():
    def __init__(self, config: TestConfig):
        self.config = config

    def test_model(self, model_generator):  # Generator
        try:
            model_generator.eval()
            random_noise = torch.randn((20, self.config.noise_size, 1, 1))
            
            with torch.no_grad():
                fake_img = model_generator(random_noise)
            
            grid_tensor = vutils.make_grid(fake_img, nrow=5, normalize=True, scale_each=True)
            grid_np = grid_tensor.cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            print(grid_np.shape)
            grid_np = (grid_np * 255).astype(np.uint8)  # Normalizasyonu kaldır, uint8 formatına çevir
            grid_pil = PIL.Image.fromarray(grid_np)
            grid_pil.save(self.config.test_img_save_path)
        
        except Exception as e:
            raise ExceptionNetwork(e, sys)
        
    def initiate_testing(self):
        
        model = load_obj(path=self.config.model_path)
        self.test_model(model_generator=model)
        
    
        
        