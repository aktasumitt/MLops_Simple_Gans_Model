import pytest
from src.pipeline.model_pipeline import ModelPipeline
from src.utils import load_obj
import torch

@pytest.fixture
def model():
    """Model oluşturma pipeline'ını çalıştırır ve model objesini döndürür."""
    model_pipeline = ModelPipeline()
    config = model_pipeline.modelconfig
    model_pipeline.run_model_creating()
    random_noise=torch.randn((1,config.noise_size,1,1))
    return config,random_noise, load_obj("callbacks/final_model/generator_model.pth")


def test_model(model):
    """Modelin doğru çalışıp çalışmadığını test eder."""
    config,random_noise, model = model

    # Test verisini modele veriyoruz
    model.eval()
    generated_img = model(random_noise)
    
    # Modelin çıktısının shape'ini test et
    assert generated_img.shape[2]==config.img_size
    assert generated_img.shape[0]==1 # batch_size
    assert generated_img.shape[1]==config.channel_size