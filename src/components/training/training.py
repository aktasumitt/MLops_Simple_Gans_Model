from src.components.training.training_module import TrainingModule
from src.components.training.validation_module import validation_model
from src.utils import (
    save_as_json, load_obj, load_checkpoints,
    save_checkpoints, save_obj
)
from torch.utils.data import DataLoader
import torch
from src.logger import logger
from src.exception.exception import ExceptionNetwork, sys
import mlflow
import subprocess
from src.entity.config_entity import TrainingConfig

import dagshub
dagshub.init(repo_owner='umitaktas', repo_name='MLops_Simple_Gans_Model', mlflow=True)


class Training():
    def __init__(self, config: TrainingConfig, TEST_MODE: bool = False):
        self.config = config
        self.TEST_MODE = TEST_MODE
        self.generator_model = load_obj(self.config.generator_model_path).to(self.config.device)
        self.discriminator_model = load_obj(self.config.discriminator_model_path).to(self.config.device)
        self.optimizer_gen = torch.optim.Adam(self.generator_model.parameters(), lr=self.config.learning_rate, betas=self.config.betas)
        self.optimizer_disc = torch.optim.Adam(self.discriminator_model.parameters(), lr=self.config.learning_rate, betas=self.config.betas)

    def load_object(self):
        try:
            train_dataset = load_obj(self.config.train_dataset_path)
            return DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        except Exception as e:
            ExceptionNetwork(e, sys)

    def load_checkpoints(self, load: bool):
        starting_epoch = 1
        if load:
            starting_epoch = load_checkpoints(
                path=self.config.checkpoint_path,
                model_disc=self.discriminator_model,
                model_gen=self.generator_model,
                optimizer_disc=self.optimizer_disc,
                optimizer_gen=self.optimizer_gen,
            )
            logger.info(f"Checkpoints loaded. Training starts from epoch {starting_epoch}.")
        return starting_epoch

    def initiate_training(self):
        try:
            result_list = []
            train_dataloader = self.load_object()
            training_module = TrainingModule(
                train_dataloader=train_dataloader,
                model_generator=self.generator_model,
                model_discriminator=self.discriminator_model,
                optimizer_disc=self.optimizer_disc,
                optimizer_gen=self.optimizer_gen,
                batch_size=self.config.batch_size,
                noise_dim=self.config.noise_dim,
                device=self.config.device
            )
            
            starting_epoch = self.load_checkpoints(load=False)   
            
            for epoch in range(starting_epoch, self.config.epochs+1):
                loss_gen, loss_disc = training_module.batch_training()
                
                save_checkpoints(
                    save_path=self.config.checkpoint_path,
                    epoch=epoch, model_gen=self.generator_model,
                    model_disc=self.discriminator_model,
                    optimizer_gen=self.optimizer_gen,
                    optimizer_disc=self.optimizer_disc,
                )
                
                images_generated = validation_model(
                    model_generator=self.generator_model,
                    noise_size=self.config.noise_dim, 
                    device=self.config.device
                )
                
                logger.info(f"Checkpoint saved at epoch {epoch}.")
                
                metrics = {"loss_discriminator": loss_disc, "loss_generator": loss_gen}
                result_list.append(metrics)
                
                mlflow.log_metrics(metrics=metrics, step=epoch)
                mlflow.log_image(images_generated, key=f"epoch_{epoch}_batch_grid.png", step=epoch)
            
            save_as_json(data=result_list, save_path=self.config.results_save_path)
            logger.info("Training results saved as JSON.")
            
            save_obj(self.generator_model, save_path=self.config.final_generator_model_path)
            save_obj(self.discriminator_model, save_path=self.config.final_discriminator_model_path)
            logger.info("Final models saved.")
        except Exception as e:
            ExceptionNetwork(e, sys)

    def start_training_with_mlflow(self):
        try:
            uri = "https://dagshub.com/umitaktas/MLops_Simple_Gans_Model.mlflow"
            mlflow.set_tracking_uri(uri=uri)
            logger.info(f"MLflow tracking started on {uri}.")
            
            mlflow.set_experiment("MLFLOW MyFirstExperiment")
            params = {
                "Batch_size": self.config.batch_size, "Learning_rate": self.config.learning_rate,
                "Betas": self.config.betas, "Epoch": self.config.epochs
            }
            
            
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.set_tag("Pytorch Training Info", "Image classification training")
                self.initiate_training()
                logger.info("Training completed. Metrics, parameters, and model saved in MLflow.")
        except Exception as e:
            ExceptionNetwork(e, sys)

if __name__ == "__main__":
    config = TrainingConfig()
    training = Training(config)
    training.start_training_with_mlflow()
