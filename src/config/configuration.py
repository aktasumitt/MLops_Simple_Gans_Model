from src.entity.config_entity import (TestConfig,
                                      TrainingConfig,
                                      DataIngestionConfig,
                                      ModelIngestionConfig,
                                      DataTransformationConfig,
                                      PredictionConfig)

from src.constants.config import Config
from src.constants.params import Params
from src.constants.schema import Schema



class Configuration():

    def __init__(self):

        self.config = Config
        self.params = Params
        self.schema = Schema

    def data_ingestion_config(self):

        configuration = DataIngestionConfig(train_data_path=self.config.train_data_path)

        return configuration

    def data_transformation_config(self):

        configuration = DataTransformationConfig(train_data_path=self.config.train_data_path,
                                                 train_dataset_save_path=self.config.train_dataset_save_path,
                                                 normalize_mean=self.params.normalize_mean,
                                                 normalize_std=self.params.normalize_std)

        return configuration

    def model_config(self):

        configuration = ModelIngestionConfig(generator_save_path=self.config.generator_save_path,
                                             discriminator_save_path=self.config.discriminator_save_path,
                                             channel_size=self.params.channel_size,
                                             noise_size=self.params.noise_dim,
                                             embed_size=self.params.embed_size,
                                             label_size=self.params.label_size,
                                             img_size=self.params.img_size)

        return configuration

    def training_config(self):

        configuration = TrainingConfig(generator_model_path=self.config.generator_save_path,
                                       discriminator_model_path=self.config.discriminator_save_path,
                                       train_dataset_path=self.config.train_dataset_save_path,
                                       checkpoint_path=self.config.checkpoint_save_path,
                                       final_generator_model_path=self.config.final_generator_model_path,
                                       final_discriminator_model_path=self.config.final_discriminator_model_path,
                                       results_save_path=self.config.results_save_path,
                                       batch_size=self.params.batch_size,
                                       noise_dim=self.params.noise_dim,
                                       device=self.params.device,
                                       learning_rate=self.params.learning_rate,
                                       betas=self.params.betas,
                                       epochs=self.params.epochs,
                                       labels=self.schema.labels
                                       )

        return configuration

    def test_config(self):

        configuration = TestConfig(noise_size=self.params.noise_dim,
                                   test_img_save_path=self.config.test_img_save_path,
                                   model_path=self.config.final_generator_model_path,
                                   device=self.params.device,
                                   labels=self.schema.labels)
        return configuration

    def prediction_config(self):
        
        configuration = PredictionConfig(noise_size=self.params.noise_dim,
                                         predicted_img_save_path=self.config.predicted_img_save_path,
                                         model_path=self.config.final_generator_model_path,
                                         device=self.params.device,
                                         labels=self.schema.labels
                                        )

        return configuration
    