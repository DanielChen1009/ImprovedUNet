import torch
import torch.nn as nn

if __name__ == '__main__':
    channel_shuffle = nn.ChannelShuffle(2)
    input = torch.randn(1, 6, 2, 2)
    print(input)
    print(channel_shuffle(input))

