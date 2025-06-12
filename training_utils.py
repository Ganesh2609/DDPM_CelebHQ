import torch
from torch import nn


class DiffusionHelper(nn.Module):

    def __init__(self, timesteps:int, schedule:str='linear', beta_start:float=0.0001, beta_end:float=0.02, device:torch.device='cpu'):
        
        super(DiffusionHelper, self).__init__()
        self.device = device
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = None
        if schedule == 'linear':
            self._initialize_linear_schedule()
        elif schedule == 'cosine':
            self._initialize_cosine_schedule()

        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.one_minus_alpha_cum_prod = 1 - self.alpha_cum_prod
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(self.one_minus_alpha_cum_prod)

    def _initialize_linear_schedule(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps).to(self.device)
    
    def _initialize_cosine_schedule(self):
        # def alpha_bar(t):
        #     return torch.cos((t / self.timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
        # alpha_bars = torch.tensor([alpha_bar(t) for t in range(self.timesteps + 1)])
        # self.betas = torch.zeros(self.timesteps)
        # for t in range(self.timesteps):
        #     self.betas[t] = 1 - alpha_bars[t + 1] / alpha_bars[t]
        # self.betas = torch.clamp(self.betas, 0.0001, 0.9999).to(self.device)
        s = 0.008
        def alpha_bar(t):
            t_tensor = torch.tensor(t, dtype=torch.float32)
            s_tensor = torch.tensor(s, dtype=torch.float32)
            timesteps_tensor = torch.tensor(self.timesteps, dtype=torch.float32)
            f_t = torch.cos((t_tensor / timesteps_tensor + s_tensor) / (1 + s_tensor) * torch.pi / 2) ** 2
            f_0 = torch.cos((s_tensor) / (1 + s_tensor) * torch.pi / 2) ** 2
            return (f_t / f_0).item()
        alpha_bars = torch.tensor([alpha_bar(t) for t in range(self.timesteps + 1)])
        self.betas = torch.zeros(self.timesteps)
        for t in range(self.timesteps):
            self.betas[t] = 1 - alpha_bars[t + 1] / alpha_bars[t]
        self.betas = torch.clamp(self.betas, 0.0001, 0.9999).to(self.device)

    def add_noise(self, x0, t):

        B, C, H, W = x0.shape
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(B, 1, 1, 1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(B, 1, 1, 1)
        noise = torch.randn_like(x0)
        return (sqrt_alpha_cum_prod * x0) + (sqrt_one_minus_alpha_cum_prod * noise), noise
    
    def get_prev_timestep(self, xt, noise_pred, t):

        B, _, _ , _ = xt.shape
        scaled_noise = self.betas[t].reshape(B, 1, 1, 1) * noise_pred / self.sqrt_one_minus_alpha_cum_prod[t].reshape(B, 1, 1, 1)
        mean = (xt - scaled_noise) / self.sqrt_alphas[t].reshape(B, 1, 1, 1)

        is_final_step = (t == 0)
        
        variance = self.betas[t].reshape(B, 1, 1, 1) * self.one_minus_alpha_cum_prod[t-1].reshape(B, 1, 1, 1) / self.one_minus_alpha_cum_prod[t].reshape(B, 1, 1, 1)
        sigma = torch.sqrt(variance)
        z = torch.randn_like(xt)
        noise_term = sigma * z
        
        result = torch.where(
            is_final_step.reshape(B, 1, 1, 1),
            mean,                             
            mean + noise_term   
        )
        
        return result
        
    def denormalize(self, image):

        return (image * 0.5) + 0.5