import torch.nn as nn
from diffusers.utils import logging
import torch

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiscriminatorHead(nn.Module):

    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        inner_channel = 1024
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(
                inplace=True
            ),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(
                inplace=True
            ),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )

        self.conv_out = nn.Conv2d(inner_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        if x.dim()==5:
            (b,ch,t,h,w) = x.shape
            # x = 
            x = x.permute(0, 2, 1, 3, 4)  # 调整为 (1, 3, 16, 98, 160)
            x = x.reshape(1*3, 16, 98, 160)  # 合并中间维度为 (3, 16, 98, 160)
            pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)  # 2x2 分块
            x = pixel_unshuffle(x)  # 输出形状 (3, 16 * 2 * 2, 98 / 2, 160 / 2)
            x = x.unsqueeze(0) # (1,3,16*2*2,49,80)
            x = x.permute((0,1,3,4,2)) # (1,3,49,80,16*4)
            x = x.reshape((b,-1,ch*4))
        # wan 15 torch.Size([1, 32760, 1536])
        
        b, twh, c = x.shape
        t = twh // (30 * 52)
        x = x.view(-1, 30 * 52, c)
        x = x.permute(0, 2, 1)
        x = x.view(b * t, c, 30, 52)
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
        return x



class Discriminator(nn.Module):

    def __init__(
        self,
        stride=8,
        num_h_per_head=1,
        adapter_channel_dims=[1536],
        total_layers=48,
    ):
        super().__init__()
        adapter_channel_dims = adapter_channel_dims * (total_layers // stride)
        self.stride = stride
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.ModuleList([
            nn.ModuleList([
                DiscriminatorHead(adapter_channel)
                for _ in range(self.num_h_per_head)
            ]) for adapter_channel in adapter_channel_dims
        ])

    def forward(self, features):
        outputs = []
        def create_custom_forward(module):

            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        assert len(features) == len(self.heads)
        for i in range(0, len(features)):
            for h in self.heads[i]:
                out = h(features[i])
                outputs.append(out)
        return outputs
