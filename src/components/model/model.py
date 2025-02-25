from src.components.model.discriminator import Discriminator
from src.components.model.generator import Generator
from src.utils import save_obj
from src.logger import logger
from src.entity.config_entity import ModelIngestionConfig


class ModelIngestion():
    
    def __init__(self, config: ModelIngestionConfig):
        self.config = config
        self.generator = Generator(
            channel_size=self.config.channel_size,
            noise_size=self.config.noise_size,
            
        )
        self.discriminator = Discriminator(
            channel_size=self.config.channel_size,
            img_size=self.config.img_size,
        )
        
    def initiate_and_save_model(self):
        save_obj(self.generator, self.config.generator_save_path)
        save_obj(self.discriminator, self.config.discriminator_save_path)
        logger.info("Generator ve Discriminator modelleri artifacts içerisine kaydedildi")
        
if __name__ == "__main__":
    config = ModelIngestionConfig()
    model_ingestion = ModelIngestion(config)
    model_ingestion.initiate_and_save_model()

