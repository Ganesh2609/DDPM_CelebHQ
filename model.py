from model_utils import *


class Unet(nn.Module):
    
    def __init__(self, num_channels:int=3, down_channels:list=[64, 128, 256, 512], mid_channels:list=[512, 512, 256], t_emb_dim:int=1024):
        
        super(Unet, self).__init__()
        self.time_embedding = TimeEmbedding(time_dim=t_emb_dim//4, emb_dim=t_emb_dim)
        up_channels = list(reversed(down_channels))
        
        self.conv_in = nn.Conv2d(in_channels=num_channels, out_channels=down_channels[0], kernel_size=3, stride=1, padding=1)

        self.downs = nn.ModuleList([])
        for i in range((len(down_channels) - 1)):
            self.downs.append(DownBlock(in_channels=down_channels[i], out_channels=down_channels[i+1], t_emd_dim=t_emb_dim, num_heads=4))
        
        self.mids = nn.ModuleList([])
        for i in range(len(mid_channels) - 1):
            self.mids.append(BottleNeckBlock(in_channels=mid_channels[i], out_channels=mid_channels[i+1], t_emd_dim=t_emb_dim, num_heads=4))

        self.ups = nn.ModuleList([])
        for i in range(1, len(up_channels)):
            self.ups.append(UpBlock(in_channels=up_channels[i]*2, out_channels=up_channels[i+1] if i!=(len(up_channels)-1) else down_channels[0], t_emd_dim=t_emb_dim, num_heads=4))

        self.norm_out = nn.GroupNorm(num_groups=8, num_channels=down_channels[0])
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(in_channels=down_channels[0], out_channels=num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        
        x = self.conv_in(x)
        t_emb = self.time_embedding(t)

        down_outs = []
        for down in self.downs:
            down_outs.append(x)
            x = down(x, t_emb)

        for mid in self.mids:
            x = mid(x, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            x = up(x, down_out, t_emb)

        x = self.norm_out(x)
        x = self.silu(x)
        x = self.conv_out(x)

        return x