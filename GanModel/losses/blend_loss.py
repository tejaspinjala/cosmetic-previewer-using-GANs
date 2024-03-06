import torch
import PIL
import os
from losses import masked_lpips


class BlendLossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(BlendLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[1.0, 'face'], [1.0, 'hair']]
        if opt.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False

        self.percept = masked_lpips.PerceptualLoss(
            model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=use_gpu
        )
        self.percept.eval()

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def _loss_percept(self, gen_im, ref_im, mask, **kwargs):
        return self.percept(gen_im, ref_im, mask=mask)

    def forward(self, gen_im, im, mask):
        loss = 0
        losses = {}
        for weight, loss_type in self.parsed_loss:
            var_dict = {
                'gen_im': gen_im,
                'ref_im': im,
                'mask': mask
            }
            tmp_loss = self._loss_percept(**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight*tmp_loss
        return loss, losses

    def cross_entropy_loss(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy(down_seg, target_mask)
        return loss