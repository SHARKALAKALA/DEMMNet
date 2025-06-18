import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MscCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', gate_gt=None):
        super(MscCrossEntropyLoss, self).__init__()

        self.weight = weight
        self.gate_gt = gate_gt
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)

            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                                    ignore_index=self.ignore_index, reduction=self.reduction)

        return loss / len(input)


if __name__ == '__main__':
    x = torch.randn(2, 2)
    print(x)
    out = x.mean(1)
    # import torch
    # ll = 'layer3_1 '
    # out = ll.split('_1')[0]+ll.split('_1')[1]
    print(out)
    # depth = torch.randn(6,3,480,640)
    # score = torch.Tensor(6,1)
    # print(score.shape)
    # print(score)
    # score = score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640)
    # # out = torch.mul(depth,score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640))
    # print(score.shape)
    # print(score)
    # torch.randn(6,3,480,640)
    # print(out)
    # out = out.view(3,480,640)
    # print(out)

    # predict = torch.randn((2, 21, 512, 512))
    # gt = torch.randint(0, 255, (2, 512, 512))

    # loss_function = MscCrossEntropyLoss()
    # result = loss_function(predict, gt)
    # print(result)
