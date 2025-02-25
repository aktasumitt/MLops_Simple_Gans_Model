from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str 
    

@dataclass
class DataTransformationConfig:
    train_data_path: Path 
    train_dataset_save_path: Path 
    normalize_mean: tuple 
    normalize_std: tuple 
    
    
@dataclass
class ModelIngestionConfig:
    generator_save_path: str 
    discriminator_save_path: str 
    channel_size: int
    noise_size: int 
    label_size: int
    embed_size: int 
    img_size: int

    
@dataclass
class TrainingConfig:
    generator_model_path: str 
    discriminator_model_path: str 
    train_dataset_path: str 
    checkpoint_path: str 
    final_generator_model_path: str 
    final_discriminator_model_path: str 
    results_save_path: str 
    batch_size: int 
    noise_dim: int 
    device: str 
    learning_rate: float 
    betas: tuple 
    epochs: int
    labels: dict
 
    
@dataclass
class TestConfig:
    noise_size: int 
    test_img_save_path: Path 
    model_path: Path 
    device: str 
    labels: dict



@dataclass
class PredictionConfig:
    noise_size: int 
    predicted_img_save_path: Path 
    model_path: Path 
    device: str 
    labels: dict
