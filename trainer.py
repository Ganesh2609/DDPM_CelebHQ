import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Tuple
from logger import TrainingLogger
from tqdm import tqdm
import sys
from torchvision.utils import save_image
from training_utils import DiffusionHelper
import time


class ModularTrainer:


    def __init__(self,
                 model: torch.nn.Module,
                 diffusion_helper: DiffusionHelper,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 log_path: Optional[str] = './train_data/logs/training.log',
                 checkpoint_path: Optional[str] = './train_data/checkpoints',
                 results_path: Optional[str] = './train_data/results',
                 num_epochs: Optional[int] = 16,
                 image_size: Optional[int] = 128,
                 num_channels: Optional[int] = 3,
                 verbose: Optional[bool] = True,
                 device: Optional[str] = None) -> None:
    
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        self.sample_images_path = results_path
        self.loss_plot_path = os.path.join(results_path, 'loss_plot.png')

        self.logger = TrainingLogger(log_path=log_path)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.diffusion_helper = diffusion_helper
        self.diffusion_helper.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.image_size = image_size
        self.num_channels = num_channels

        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer = optimizer or torch.optim.Adam(params=self.model.parameters(), lr=1e-4)
        self.scheduler = scheduler

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.plot_update_step = 500
        self.sample_generation_step = 3500

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = float('inf')
        self.history: Dict[str, list] = {'Training Loss': [], 'Testing Loss': []}
        self.step_history: Dict[str, list] = {'Training Loss': [], 'Testing Loss': []}


    def update_loss_plot(self) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle("Loss Metrics")

        train_loss_data = self.step_history['Training Loss']
        if train_loss_data:
            train_x_coords = [(i + 1) * self.plot_update_step for i in range(len(train_loss_data))]
            axes[0].plot(train_x_coords, train_loss_data, color='blue', label='Train Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Global Training Steps')
        axes[0].set_ylabel('Loss')

        if self.test_loader:
            test_loss_data = self.step_history['Testing Loss']
            if test_loss_data:
                test_x_coords = [(i + 1) * self.plot_update_step for i in range(len(test_loss_data))]
                axes[1].plot(test_x_coords, test_loss_data, color='orange', label='Test Loss')
            axes[1].set_title('Testing Loss')
            axes[1].set_xlabel('Global Training Steps')
            axes[1].set_ylabel('Loss')
        else:
            axes[1].set_title('Testing Loss (No Test Loader)')
            axes[1].text(0.5, 0.5, 'No test data loader provided.', ha='center', va='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.loss_plot_path)
        plt.close(fig)

        return


    @torch.no_grad()
    def generate_and_save_samples(self, num_samples: int = 8) -> None:

        self.model.eval()
        xt = torch.randn((num_samples, self.num_channels, self.image_size, self.image_size), device=self.device)

        for t_val in reversed(range(self.diffusion_helper.timesteps)):
            t = torch.full((num_samples,), t_val, device=self.device, dtype=torch.long)
            noise_pred = self.model(xt, t)
            xt = self.diffusion_helper.get_prev_timestep(xt=xt, noise_pred=noise_pred, t=t)
        
        generated_images = self.diffusion_helper.denormalize(xt.cpu())
        plot_path = os.path.join(self.sample_images_path, f"Epoch_{self.current_epoch}.png")
        #plot_path = os.path.join(self.sample_images_path, f"Epoch_trial.png")
        save_image(generated_images, plot_path, nrow=4)

        self.model.train()

        return


    def train_epoch(self) -> None:

        self.model.train()
        total_train_loss = 0.0
        
        num_diffusion_timesteps = self.diffusion_helper.timesteps

        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)')
        
        for i, images in progress_bar: 
            
            images = images.to(self.device) 
            t = torch.randint(0, num_diffusion_timesteps, (images.shape[0],), device=self.device).long()
            noisy_images_xt, actual_noise = self.diffusion_helper.add_noise(images, t)

            noise_pred = self.model(noisy_images_xt, t)
            
            loss = self.loss_fn(noise_pred, actual_noise)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_train_loss += loss.item()
            
            if (self.current_step - 1 ) % self.sample_generation_step == 0:
                self.generate_and_save_samples()

            if self.current_step % self.plot_update_step == 0: 
                self.step_history['Training Loss'].append(loss.item())
                self.update_loss_plot()
                time.sleep(30)


            lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Training Loss': f'{total_train_loss / (i + 1):.4f}',
                'Learning Rate': f'{lr:.2e}'
            })
            self.current_step += 1

        avg_train_loss = total_train_loss / len(self.train_loader)
        self.history['Training Loss'].append(avg_train_loss)
        self.logger.info(f"Training Loss for Epoch {self.current_epoch}: {avg_train_loss:.4f}")

        return


    def test_epoch(self) -> Optional[float]:
            
        self.model.eval()
        total_test_loss = 0.0
        num_diffusion_timesteps = self.diffusion_helper.timesteps

        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Testing)', file=sys.stdout)

        with torch.no_grad():
            for i, images in progress_bar:

                images = images.to(self.device)
                t = torch.randint(0, num_diffusion_timesteps, (images.shape[0],), device=self.device).long()
                noisy_images_xt, actual_noise = self.diffusion_helper.add_noise(images, t)

                noise_pred = self.model(noisy_images_xt, t)
            
                loss = self.loss_fn(noise_pred, actual_noise)
                total_test_loss += loss.item()

                if self.current_step % self.plot_update_step == 0: 
                     self.step_history['Testing Loss'].append(loss.item())

                progress_bar.set_postfix({
                    'Batch Loss': f'{loss.item():.4f}',
                    'Testing Loss': f'{total_test_loss / (i + 1):.4f}'
                })

        avg_test_loss = total_test_loss / len(self.test_loader)
        self.history['Testing Loss'].append(avg_test_loss)
        self.logger.info(f"Epoch {self.current_epoch} Average Testing Loss: {avg_test_loss:.4f}\n")

        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_test_loss)
            else:
                self.scheduler.step()

        if avg_test_loss < self.best_metric:
            self.best_metric = avg_test_loss
            self.save_checkpoint(is_best=True)
        
        return 
    

    def train(self, resume_from: Optional[str] = None) -> None:

        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            self.logger.log_training_resume(
                epoch=self.current_epoch,
                global_step=self.current_step,
                total_epochs=self.num_epochs
            )
        else:
            self.logger.info(f"Starting training for {self.num_epochs} epochs from scratch.")

        self.logger.info(f"Target epochs: {self.num_epochs}. Starting from epoch: {self.current_epoch}")

        for epoch in range(self.current_epoch, self.num_epochs + 1):

            self.current_epoch = epoch
            self.train_epoch()
            
            if self.test_loader:
                self.test_epoch()

            self.save_checkpoint() 

        return


    def save_checkpoint(self, is_best: Optional[bool] = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'step_history': self.step_history,
            'best_metric': self.best_metric,
            'image_size': self.image_size,
            'num_channels': self.num_channels
        }

        if is_best:
            path = os.path.join(self.checkpoint_path, 'best_model.pth')
        else:
            path = os.path.join(self.checkpoint_path, f'model_epoch_{self.current_epoch}.pth')

        torch.save(checkpoint, path)

        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} (Epoch {self.current_epoch}) saved to {path}")

        return


    def load_checkpoint(self, checkpoint_path_to_load: str):

        if not os.path.exists(checkpoint_path_to_load):
            self.logger.error(f"Checkpoint file not found: {checkpoint_path_to_load}")
            return

        checkpoint = torch.load(checkpoint_path_to_load, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.scheduler is None:
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0) + 1 
        self.current_step = checkpoint.get('current_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.image_size = checkpoint.get('image_size', self.image_size)
        self.num_channels = checkpoint.get('num_channels', self.num_channels)
            
        loaded_history = checkpoint.get('history', {})
        self.history = loaded_history
            
        loaded_step_history = checkpoint.get('step_history', {})
        self.step_history = loaded_step_history
            
        self.logger.info(f"Loaded checkpoint from {checkpoint_path_to_load}")

        return

