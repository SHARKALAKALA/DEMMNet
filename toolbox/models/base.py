import torch
import torch.nn as nn
from toolbox.models.decoders import Unet, light
from toolbox.models.encoder.mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = mit_b2()

    def forward(self, rgb):
        fea_list = self.rgb(rgb)
        return fea_list


class decoder(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512]):
        super().__init__()
        self.de = light.lightdecoder(inc=in_channels)

    def forward(self, fea_lilst):
        fin = self.de(fea_lilst)
        return fin


class seg_seg(nn.Module):
    def __init__(self, f_c=[64, 128, 320, 512], num_classes=41):
        super(seg_seg, self).__init__()
        # 32, 64, 160, 256   64, 128, 320, 512  64, 128, 256, 512  48, 96, 192, 384  40, 80, 160, 320
        self.en = encoder()
        self.de = decoder(in_channels=f_c)
        self.fin = nn.Conv2d(f_c[0], num_classes, 3, 1, 1)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, rgb):
        fea_list = self.en(rgb)
        seg = self.de(fea_list)
        pred = self.up4(self.fin(seg))

        return pred


if __name__ == '__main__':
    a = torch.randn(2, 3, 480, 640).cuda()
    b = torch.randn(2, 1, 480, 640).cuda()
    model = seg_seg()
    model.cuda()
    s = model(a)
    for i in s:
        print(i.shape)
