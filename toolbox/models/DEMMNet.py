import torch
import torch.nn as nn
from toolbox.models.encoder.mix_transformer import mit_b5, mit_b4, mit_b3, mit_b2, mit_b1
import torch.nn.functional as F


class cbr(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size, padding=0, dilation=1, stride=1, groups=1,
                 activation=nn.ReLU(inplace=True)):
        super(cbr, self).__init__()
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride,
                                          groups=groups))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)


class dse(nn.Module):
    def __init__(self, c):
        super(dse, self).__init__()
        self.v = nn.Conv2d(c, c, 3, 1, 1)
        self.k = nn.Conv2d(c, c, 3, 1, 1)
        self.q = nn.Sequential(
            cbr(1, 1, 1),
            cbr(1, 1, 3, 1, 1),
            cbr(1, 1, 5, 2, 1),
            cbr(1, 1, 7, 3, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, f, d):
        q = self.q(d)
        k = self.k(f)
        v = self.v(f)
        f = self.sig(q * k) * v + v
        return f


class cpss(nn.Module):
    def __init__(self, num_classes, inc1, inc2, block_ratio=5):
        super().__init__()
        self.fea_seg = nn.Sequential(
            nn.Conv2d(inc1, inc1, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Tanh()
        )
        self.fea_class = cbr(inc2, inc1, 3, 1, 1)
        self.class_head = nn.Sequential(
            nn.Conv2d(inc1, num_classes, 1),
            nn.BatchNorm2d(num_classes),
        )
        self.num_classes = num_classes
        self.finc = inc1
        self.block_ratio = block_ratio

    def forward(self, fea_seg, fea_class):
        fea_aff = self.fea_seg(fea_seg)
        b, C, H, W = fea_aff.shape
        block_h, block_w = H // self.block_ratio, W // self.block_ratio

        fea_class = self.fea_class(fea_class)
        b, c, h, w = fea_class.shape
        f_ratio = H // h
        down = nn.AvgPool2d(f_ratio, f_ratio).cuda()
        
        num_blocks_h = self.block_ratio
        num_blocks_w = self.block_ratio
        
        fea_aff_blocks = fea_aff.view(
            b, C, num_blocks_h, block_h, num_blocks_w, block_w
        )
        fea_aff_blocks = fea_aff_blocks.permute(0, 2, 4, 1, 3, 5).contiguous()
        block_pixels = block_h * block_w
        fea_aff_blocks = fea_aff_blocks.view(b, num_blocks_h, num_blocks_w, C, block_pixels)
        fea_aff_blocks_t = fea_aff_blocks.permute(0, 1, 2, 4, 3)
        affinity_blocks = torch.matmul(fea_aff_blocks_t, fea_aff_blocks)

        fea_class_blocks = fea_class.view(
            b, self.finc, num_blocks_h, block_h//f_ratio, num_blocks_w, block_w//f_ratio
        )
        fea_class_blocks = fea_class_blocks.permute(0, 2, 4, 1, 3, 5).contiguous()
        fea_class_blocks = fea_class_blocks.view(b, num_blocks_h, num_blocks_w, self.finc, block_pixels//f_ratio**2)
        fea_class_blocks = fea_class_blocks.permute(0, 1, 2, 4, 3)
        
        masked_fea = torch.matmul(
                    down(
                        affinity_blocks.view(
                            b, num_blocks_h, num_blocks_w, block_pixels, block_h, block_w
                            ).view(-1, 1, block_h, block_w)
                        ).view(
                            b, num_blocks_h, num_blocks_w, block_pixels, block_h//f_ratio, block_w//f_ratio
                        ).flatten(-2),
                    fea_class_blocks
                ).view(b, num_blocks_h, num_blocks_w, block_h, block_w, self.finc)
        masked_fea = masked_fea.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, H, W, self.finc)
        cpss_seg = self.class_head(masked_fea.permute(0, 3, 1, 2))

        return cpss_seg, affinity_blocks, (b, self.num_classes, H, W)


class depth_flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = [nn.AvgPool2d(s) for s in [4, 8, 16, 32]]

    def forward(self, depth):
        depth_down = [down(depth) for down in self.down]
        return depth_down


class fea_fuse(nn.Module):
    def __init__(self, fea_channel):
        super(fea_fuse, self).__init__()
        self.dse1 = dse(fea_channel[0])
        self.dse2 = dse(fea_channel[1])
        self.dse3 = dse(fea_channel[2])
        self.dse4 = dse(fea_channel[3])

    def forward(self, fea_list, d):
        fea_list[0] = self.dse1(fea_list[0], d[0])
        fea_list[1] = self.dse2(fea_list[1], d[1])
        fea_list[2] = self.dse3(fea_list[2], d[2])
        fea_list[3] = self.dse4(fea_list[3], d[3])
        return fea_list


class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = mit_b2()

    def forward(self, rgb):
        fea_list = self.rgb(rgb)
        return fea_list


class decoder(nn.Module):
    def __init__(self, inc):
        super(decoder, self).__init__()
        self.conv1 = nn.Conv2d(inc[3], inc[0], 1)
        self.conv2 = nn.Conv2d(inc[2], inc[0], 1)
        self.conv3 = nn.Conv2d(inc[1], inc[0], 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, fea_list):
        _, _, h, w = fea_list[0].shape
        f1 = F.interpolate(self.conv1(fea_list[3]), size=(h, w), mode='nearest')
        f2 = F.interpolate(self.conv2(fea_list[2]), size=(h, w), mode='nearest')
        f3 = F.interpolate(self.conv3(fea_list[1]), size=(h, w), mode='nearest')
        f = f1 + f2 + f3 + fea_list[0]
        return f


class DEMMNet(nn.Module):
    def __init__(self, fea_channel=[64, 128, 320, 512], num_classes=41):
        super(DEMMNet, self).__init__()
        self.encoder = encoder()
        self.depth_flow = depth_flow()
        self.fea_fuse = fea_fuse(fea_channel)
        self.decoder = decoder(fea_channel)
        self.cpss = cpss(num_classes, fea_channel[0], fea_channel[-1], 5)
        self.seg_head = nn.Sequential(
            nn.Conv2d(fea_channel[0], num_classes, 3, 1, 1),
            nn.BatchNorm2d(num_classes),
        )
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, rgb, depth):
        fea_list = self.encoder(rgb)
        fea_class = fea_list[-1] 

        depth_list = self.depth_flow(depth)
        fused_fea_list = self.fea_fuse(fea_list, depth_list)
        fea_seg = self.decoder(fused_fea_list)
        cpss_out = self.cpss(fea_seg, fea_class)
        seg = self.seg_head(fea_seg)
        pred = self.up4(seg + F.interpolate(cpss_out[0], size=seg.shape[-2:], mode='bilinear'))

        return pred, seg, cpss_out


if __name__ == '__main__':
    from thop import profile
    import time
    a = torch.randn(1, 3, 480, 640).cuda()
    b = torch.randn(1, 1, 480, 640).cuda()
    model = DEMMNet()
    model.cuda()
    model.eval()
    with torch.no_grad():
        s = model(a, b)
        print(s[0].shape)
        print(s[1].shape)
        print(s[2][0].shape, s[2][1].shape)
