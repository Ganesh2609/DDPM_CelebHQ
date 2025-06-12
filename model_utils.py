import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int):

        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:

        factor = 10000 ** ( torch.arange(start=0, end=self.dim//2, device=time.device) / (self.dim//2) )
        t_emb = time[:, None].repeat(1, self.dim//2) / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return t_emb


class TimeEmbedding(nn.Module):

    def __init__(self, time_dim: int, emb_dim: int):

        super(TimeEmbedding, self).__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:

        time_emb = self.time_embedding(time)
        return self.mlp(time_emb)
    


class DownBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, t_emd_dim:int, num_heads:int, num_groups:int=8, down_sample:bool=True):

        super(DownBlock, self).__init__()
        self.conv_1 = self._conv_block(in_channels, out_channels)
        self.conv_2 = self._conv_block(out_channels, out_channels)
        self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.time_fc = nn.Sequential(nn.SiLU(), nn.Linear(t_emd_dim, out_channels))
        self.attention_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.down_sample = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=8):

        return nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
    
    def forward(self, x, time_emd):

        residual_input = self.residual_conv(x)
        x = self.conv_1(x)
        x = x + self.time_fc(time_emd)[:, :, None, None]
        x = self.conv_2(x)
        x = x + residual_input
        
        B, C, H, W = x.shape
        attn = self.attention_norm(x.reshape(B, C, H*W))
        attn = attn.permute(0, 2, 1)
        attn, _ = self.attention(attn, attn, attn)
        attn = attn.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + attn

        x = self.down_sample(x)
        return x
    

class BottleNeckBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, t_emd_dim:int, num_heads:int, num_groups:int=8):

        super(BottleNeckBlock, self).__init__()
        self.residual_1 = nn.ModuleList([
            self._conv_block(in_channels, out_channels),
            self._conv_block(out_channels, out_channels)
        ])
        self.residual_2 = nn.ModuleList([
            self._conv_block(out_channels, out_channels),
            self._conv_block(out_channels, out_channels)
        ])
        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        ])
        self.time_fc = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(t_emd_dim, out_channels)),
            nn.Sequential(nn.SiLU(), nn.Linear(t_emd_dim, out_channels))
        ])
        self.attention_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=8):

        return nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
    
    def forward(self, x, time_emd):

        residual_input = self.residual_conv[0](x)
        x = self.residual_1[0](x)
        x = x + self.time_fc[0](time_emd)[:, :, None, None]
        x = self.residual_1[1](x)
        x = x + residual_input
        
        B, C, H, W = x.shape
        attn = self.attention_norm(x.reshape(B, C, H*W))
        attn = attn.permute(0, 2, 1)
        attn, _ = self.attention(attn, attn, attn)
        attn = attn.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + attn

        residual_input = self.residual_conv[1](x)
        x = self.residual_2[0](x)
        x = x + self.time_fc[1](time_emd)[:, :, None, None]
        x = self.residual_2[1](x)
        x = x + residual_input

        return x
    

class UpBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, t_emd_dim:int, num_heads:int, num_groups:int=8, up_sample:bool=True):

        super(UpBlock, self).__init__()
        self.conv_1 = self._conv_block(in_channels, out_channels)
        self.conv_2 = self._conv_block(out_channels, out_channels)
        self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.time_fc = nn.Sequential(nn.SiLU(), nn.Linear(t_emd_dim, out_channels))
        self.attention_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=4, stride=2, padding=1) if up_sample else nn.Identity()
        
    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=8):

        return nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
    
    def forward(self, x, down_out, time_emd):
        
        x = self.up_sample(x)
        x = torch.cat([x, down_out], dim=1)

        residual_input = self.residual_conv(x)
        x = self.conv_1(x)
        x = x + self.time_fc(time_emd)[:, :, None, None]
        x = self.conv_2(x)
        x = x + residual_input
        
        B, C, H, W = x.shape
        attn = self.attention_norm(x.reshape(B, C, H*W))
        attn = attn.permute(0, 2, 1)
        attn, _ = self.attention(attn, attn, attn)
        attn = attn.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + attn

        return x