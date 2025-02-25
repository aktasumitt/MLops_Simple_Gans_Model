import torch 
import tqdm
from src.components.training.gradient_penalty import Gradient_penalty
from src.exception.exception import ExceptionNetwork,sys


class TrainingModule():
    def __init__(self, train_dataloader, model_generator, model_discriminator, optimizer_gen, optimizer_disc, batch_size, noise_dim, device):

        self.train_dataloader = train_dataloader
        self.model_generator = model_generator
        self.model_discriminator = model_discriminator
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.device = device
        
        self.model_generator.train()
        self.model_discriminator.train()

    def train_discriminator(self,real_img):
        try:
            for _ in range(5):

                random_noise = torch.randn((self.batch_size, self.noise_dim, 1, 1)).to(self.device)   # random noise to generate fake img
                fake_img = self.model_generator(random_noise)   # Generate Fake img


                self.model_discriminator.zero_grad()
                fake_disc = self.model_discriminator(fake_img).reshape(-1)
                real_disc = self.model_discriminator(real_img).reshape(-1)
                
                # Gradient Penalty Calculation
                Gradient_P = Gradient_penalty(discriminator_model=self.model_discriminator,
                                            fake_img=fake_img,
                                            real_img=real_img,
                                            devices=self.device)

                loss_disc = (-(torch.mean(real_disc) -torch.mean(fake_disc))+(Gradient_P*10))
                loss_disc.backward(retain_graph=True)
                self.optimizer_disc.step()

            return loss_disc.item(),fake_img
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
    def train_generator(self, fake_img):  # Generator
        try:  
            self.model_generator.zero_grad()
            fake_disc_gen = self.model_discriminator(fake_img).reshape(-1)
            gen_loss = -torch.mean(fake_disc_gen)
            gen_loss.backward()
            self.optimizer_gen.step()

            return gen_loss.item()
        
        except Exception as e:
                raise ExceptionNetwork(e,sys)
    
    def batch_training(self):
        try:
            PB = tqdm.tqdm(range(len(self.train_dataloader)), "Training Process")
            GEN_LOSS=0
            DISC_LOSS=0

            for batch, (img, _) in enumerate(self.train_dataloader):
                
                real_img = img.to(self.device)                

                
                # train generator and discriminator
                loss_disc,fake_img=self.train_discriminator(real_img)
                loss_gen=self.train_generator(fake_img)
                
                # adding losses for each batch
                GEN_LOSS+=loss_gen
                DISC_LOSS+=loss_disc
                PB.update(1)
                if batch>0 and batch%100==0:
                    PB.set_postfix({"Gen_loss":GEN_LOSS/(batch+1),
                                    "disc_loss":DISC_LOSS/(batch+1)})
                    
            
            TOTAL_GEN_LOSS=GEN_LOSS/(batch+1)
            TOTAL_DISC_LOSS=DISC_LOSS/(batch+1)
                
            return TOTAL_GEN_LOSS,TOTAL_DISC_LOSS
        except Exception as e:
            raise ExceptionNetwork(e,sys)