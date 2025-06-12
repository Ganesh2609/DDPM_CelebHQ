import torch
from torch import nn
from trainer import ModularTrainer
from dataset import get_data_loaders
from model import Unet
from training_utils import DiffusionHelper


def main():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = Unet(num_channels=3)

    img_size = 64
    diffusion_timesteps = 1024
    learning_rate = 1e-6

    data_root = "E:/Deep learning assets/Datasets/Celeba HQ"
    #data_root = "E:/Deep learning assets/Datasets/Celeba HQ - Low level Testing"
    train_loader, test_loader = get_data_loaders(root=data_root, img_size=img_size, batch_size=4)
    diffusion_helper = DiffusionHelper(timesteps=diffusion_timesteps, schedule='linear', device=device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    trainer = ModularTrainer(
        model=model,
        diffusion_helper=diffusion_helper,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        log_path='Train Data/Logs/train_1_celeb.log',
        checkpoint_path='Train Data/Checkpoints/train 1 celeb',
        results_path='Train Data/Results/train 1 celeb',
        num_epochs=64,
        image_size=img_size,
        verbose=True,
        device=device
    )

    #trainer.train()
    trainer.train(resume_from="Train Data/Checkpoints/train 1 celeb/model_epoch_46.pth")

if __name__ == '__main__':
    main()
