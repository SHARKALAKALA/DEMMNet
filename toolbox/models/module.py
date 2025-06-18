import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F

class cpssloss(nn.Module):
    def __init__(self, ce=nn.CrossEntropyLoss()):
        super(cpssloss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = ce
        self.up = nn.Upsample(scale_factor=16, mode='bilinear')

    def predict_from_block_affinity(self, mb, gt, ratio, num_classes):
        gt = gt.squeeze(1)
        b, h, w = gt.shape
        block_h = h // ratio
        block_w = w // ratio
        block_size = block_h * block_w
        
        gt_blocks = gt.view(b, ratio, block_h, ratio, block_w)
        gt_blocks = gt_blocks.permute(0, 1, 3, 2, 4)  # (b, ratio, ratio, block_h, block_w)
        gt_blocks_flat = gt_blocks.reshape(b, ratio, ratio, block_size)

        pred_map = torch.zeros_like(gt.unsqueeze(1).repeat(1, num_classes, 1, 1))
        
        for i in range(ratio):
            for j in range(ratio):
                h_start, h_end = i * block_h, (i + 1) * block_h
                w_start, w_end = j * block_w, (j + 1) * block_w

                mb_block = mb[:, i, j]  # (b, block_size, block_size)
                gt_block = gt[:, h_start:h_end, w_start:w_end]  # (b, block_h, block_w)

                gt_flat = gt_block.flatten(1)  # (b, block_size)

                if not gt_flat.dtype in (torch.long, torch.int32, torch.int64):
                    gt_flat = gt_flat.long()

                gt_one_hot = F.one_hot(gt_flat, num_classes).float()  # (b, block_size, num_classes)

                similarity = torch.cosine_similarity(
                    nn.Sigmoid()(mb_block).unsqueeze(2),  # (b, block_size, 1, block_size)
                    gt_one_hot.transpose(1, 2).unsqueeze(1),  # (b, 1, num_classes, block_size)
                    dim=-1
                )  # (b, block_size, num_classes)
                
                pred_block = similarity.transpose(1, 2).view(b, num_classes, block_h, block_w)
                pred_map[:, :, h_start:h_end, w_start:w_end] = pred_block
        
        return pred_map

    def compute_block_affinity_gt(self, gt, ratio):
        gt = gt.squeeze(1)
        b, h, w = gt.shape
        block_h = h // ratio
        block_w = w // ratio
        
        gt_blocks = gt.view(b, ratio, block_h, ratio, block_w)
        gt_blocks = gt_blocks.permute(0, 1, 3, 2, 4)
        
        gt_blocks_flat = gt_blocks.reshape(b, ratio, ratio, block_h * block_w)

        gt_expanded1 = gt_blocks_flat.unsqueeze(-1)  # (b, ratio, ratio, block_h*block_w, 1)
        gt_expanded2 = gt_blocks_flat.unsqueeze(-2)  # (b, ratio, ratio, 1, block_h*block_w)
        
        mb_gt = (gt_expanded1 == gt_expanded2).float()
        
        return mb_gt

    def forward(self, fuzzy_map, mat_size, label=None, bs=5):
        b, c, h, w = mat_size
        gt = F.interpolate(label.unsqueeze(1).float(), size=(h, w))

        gt_mat = self.compute_block_affinity_gt(gt, bs)
        loss_bce = self.bce(fuzzy_map, gt_mat)

        fuzzy_seg_gt = self.predict_from_block_affinity(fuzzy_map, gt, bs, c)
        loss_ce_gt = self.ce(fuzzy_seg_gt, label.long())
        return loss_bce, loss_ce_gt

if __name__ == '__main__':
    a = torch.randn(2, 5, 5, 192, 192).cuda()
    b = torch.randint(low=0, high=41, size=(2, 60, 80), dtype=torch.int32).cuda()
    fl = cpssloss().cuda()
    a, b = fl(a, (2, 41, 60, 80), b)
    print(a, b)
