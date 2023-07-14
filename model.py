import math

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torchvision.utils



class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, mid_channels, base_channels=4, cardinality=32, downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        agg_channels = int(math.floor(mid_channels * base_channels / 64) * cardinality)
        out_channels = mid_channels * self.expansion

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, agg_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(agg_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(agg_channels, agg_channels, kernel_size=3, stride=stride, padding=1,
                      dilation=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(agg_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(agg_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEModule(out_channels)
         )
        self.final_act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x = self.conv(x)
        x += residual
        x = self.final_act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_branch = x.mean(dim=(2, 3), keepdim=True)
        x_branch = self.conv(x_branch)
        return x * x_branch


class AsymmetricConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(AsymmetricConv, self).__init__()
        self.square_conv = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels, padding=1, kernel_size=kernel_size,
                                     stride=1)
        self.hor_conv = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=(1, kernel_size), stride=1)
        self.ver_conv = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=(kernel_size, 1), stride=1)
        self.square_bn = nn.BatchNorm2d(out_channels)
        self.hor_bn = nn.BatchNorm2d(out_channels)
        self.ver_bn = nn.BatchNorm2d(out_channels)

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b

    def forward(self, x):
        # if self.training:
            xsq = self.square_conv(x)
            xsq = self.square_bn(xsq)
            xh = self.hor_conv(x)[:, :, 1:-1]
            xh = self.hor_bn(xh)
            xv = self.ver_conv(x)[:, :, :, 1:-1]
            xv = self.hor_bn(xv)
            x = xsq + xh + xv
            return x
        # else:
            # print(self.get_equivalent_kernel_bias())
        # TODO: DO CONVOLUTION FUSION


class ShuffleAttnBlock(nn.Module):
    def __init__(self, in_channels, group_num=16):
        super(ShuffleAttnBlock, self).__init__()
        self.group_size = in_channels//group_num
        self.layer_size = self.group_size//2

        self.gn = nn.GroupNorm(self.layer_size//2, self.layer_size)
        self.conv_top = nn.Conv2d(self.layer_size, self.layer_size, kernel_size=1, bias=True)
        self.conv_bottom = nn.Conv2d(self.layer_size, self.layer_size, kernel_size=1, bias=True)
        self.conv_bottom = nn.Conv2d(self.layer_size, self.layer_size, kernel_size=1, bias=True)
        self.sig = nn.Sigmoid()
        # self.channel_shuffle = nn.ChannelShuffle(group_num)

    def forward(self, x):
        # Splits into group_num groups
        groups = x.split(self.group_size, dim=1)
        final_comb = None
        for group in groups:
            # Splits into two layers
            layers = group.split(self.layer_size, dim=1)
            l1 = layers[0].mean((2, 3), keepdim=True)
            l2 = self.gn(layers[1])
            l1 = self.conv_top(l1)
            l2 = self.conv_bottom(l2)
            l1 = self.sig(l1)
            l2 = self.sig(l2)
            l1_new = layers[0] * l1
            l2_new = layers[1] * l2
            group_new = torch.concat((l1_new, l2_new), dim=1)
            if final_comb is None:
                final_comb = group_new
            else:
                final_comb = torch.concat((final_comb, group_new), dim=1)
        # return self.channel_shuffle(final_comb)
        return final_comb


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ImprovedUNet, self).__init__()
        features = (64, 128, 256, 512)
        # SE-ResNeXt-50 stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )

        in_channels = features[0]

        # SE-ResNeXt-50 downsampling

        layers = (3, 4, 6, 3)
        self.downs = nn.ModuleList()
        for i in range(len(features)):
            layer = layers[i]
            layer_blocks = []
            layer_stride = 1 if i == 0 else 2
            for j in range(layer):
                if j == 0:
                    if layer_stride == 1:
                        layer_blocks.append(Bottleneck(in_channels, features[i], downsample=nn.Sequential(
                            nn.Conv2d(in_channels, features[i] * Bottleneck.expansion, kernel_size=1, stride=1,
                                      padding=0, bias=False),
                            nn.BatchNorm2d(features[i] * Bottleneck.expansion)
                        ), stride=layer_stride))
                    else:
                        layer_blocks.append(Bottleneck(in_channels, features[i], downsample=nn.Sequential(
                            nn.MaxPool2d(kernel_size=layer_stride, stride=layer_stride),
                            nn.Conv2d(in_channels, features[i] * Bottleneck.expansion, kernel_size=1, stride=1,
                                      padding=0, bias=False),
                            nn.BatchNorm2d(features[i] * Bottleneck.expansion)
                        ), stride=layer_stride))

                else:
                    layer_blocks.append(Bottleneck(in_channels, features[i]))
                in_channels = features[i] * Bottleneck.expansion
            self.downs.append(nn.Sequential(*layer_blocks))

        # Asymmetric Conv Block Upsampling
        sf = [4, 2]
        self.ups_pre = nn.ModuleList()
        self.ups_post = nn.ModuleList()
        self.shuffles = nn.ModuleList()
        self.esc_convs = nn.ModuleList()
        self.esc_convs.append(nn.ModuleList())
        self.esc_convs.append(nn.ModuleList())
        for i in range(len(features)):
            feature = features[len(features) - i - 1]
            if i == 0:
                in_feature = feature * 2
            else:
                in_feature = feature * 2

            self.ups_pre.append(nn.ConvTranspose2d(in_feature, feature, kernel_size=2, stride=2))
            self.ups_pre.append(nn.Sequential(
                nn.Conv2d(in_feature, feature, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                AsymmetricConv(feature, feature, kernel_size=3)
            ))

            self.ups_post.append(nn.ConvTranspose2d(in_feature, feature, kernel_size=2, stride=2))
            self.ups_post.append(nn.Sequential(
                nn.Conv2d(in_feature, feature, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                AsymmetricConv(feature, feature, kernel_size=3)
            ))

            self.shuffles.append(ShuffleAttnBlock(feature))
            if i == 1:
                self.esc_convs[0].append(nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=sf[i]),
                    nn.Conv2d(feature, 64, kernel_size=3, stride=1, bias=False)
                ))
                self.esc_convs[1].append(nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=sf[i]),
                    nn.Conv2d(feature, 64, kernel_size=3, stride=1, bias=False)
                ))
            if i == 0:
                self.esc_convs[0].append(nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=sf[i]),
                    nn.Conv2d(feature, 64, kernel_size=1, stride=1, bias=False)
                ))
                self.esc_convs[1].append(nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=sf[i]),
                    nn.Conv2d(feature, 64, kernel_size=1, stride=1, bias=False)
                ))

        self.final_convs = nn.ModuleList()
        self.final_convs.append(nn.Sequential(
            nn.Conv2d(4 * features[0], features[0], kernel_size=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        ))
        self.final_convs.append(nn.Sequential(
            nn.Conv2d(4 * features[0], features[0], kernel_size=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        ))
        self.to_output = nn.Sequential(
            nn.Conv2d(2 * features[0], out_channels, kernel_size=1),
        )

    def forward(self, pre, post):
        # Downsampling
        skip_conn_pre = []
        skip_conn_post = []
        extra_skip_conn = []
        pre = self.stem(pre)
        skip_conn_pre.append(pre)
        post = self.stem(post)
        skip_conn_post.append(post)
        for down in self.downs[:-1]:
            pre = down(pre)
            skip_conn_pre.append(pre)
            post = down(post)
            skip_conn_post.append(post)

        pre = self.downs[-1](pre)
        post = self.downs[-1](post)
        skip_conn_post = skip_conn_post[::-1]
        skip_conn_pre = skip_conn_pre[::-1]
        # print(f"Bottom: {pre.shape}")

        # Upsampling
        for i in range(4):

            pre = self.ups_pre[2 * i](pre)

            if pre.shape != skip_conn_pre[i].shape:
                skip_conn_pre[i] = T.CenterCrop(size=pre.shape[2:])(skip_conn_pre[i])
            pre = torch.cat((skip_conn_pre[i], pre), dim=1)

            post = self.ups_post[2 * i](post)
            if pre.shape != skip_conn_pre[i].shape:
                skip_conn_post[i] = T.CenterCrop(size=post.shape[2:])(skip_conn_post[i])
            post = torch.cat((skip_conn_pre[i], post), dim=1)

            # print(f"Upsample: {i} {pre.shape}")

            pre = self.ups_pre[2 * i + 1](pre)
            post = self.ups_post[2 * i + 1](post)
            # print(i, pre.shape)
            # print(f"Conv: {i} {pre.shape}")

            pre = post = self.shuffles[i](pre + post)
            if i <= 1:
                extra_skip_conn.append([self.esc_convs[0][i](pre), self.esc_convs[1][i](post)])
            if i == 3:
                resize = T.Resize(pre.shape[2:])
                pre = torch.concat((resize(extra_skip_conn[0][0]),
                                    resize(extra_skip_conn[1][0]),
                                    resize(skip_conn_pre[i]), pre), dim=1)
                post = torch.concat((resize(extra_skip_conn[0][1]),
                                     resize(extra_skip_conn[1][1]),
                                     resize(skip_conn_post[i]), post), dim=1)
        pre = self.final_convs[0](pre)
        post = self.final_convs[1](post)
        combined = torch.concat((pre, post), dim=1)
        return self.to_output(combined)


if __name__ == '__main__':
    mod = ImprovedUNet(3, 5).to(device=torch.device("mps"))
    mod.eval()
    pr = torch.randn(4, 3, 512, 512).to(device=torch.device("mps"))
    pos = torch.randn(4, 3, 512, 512).to(device=torch.device("mps"))
    out = mod(pr, pos)
