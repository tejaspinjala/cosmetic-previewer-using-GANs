import torch
from torch import nn
from torch.nn import functional as F


class CustomLoss(nn.Module):
    def __init__(self, loss_type="Loss2"):
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, x, y, mask=None, include_bkgd=True):
        if include_bkgd:
            return self.mse_loss(x, y) if self.loss_type == "Loss2" else F.l1_loss(x, y)

        Ax, Cx, Hx, Wx = x.shape
        Am, Cm, Hm, Wm = mask.shape
        mask = self.set_mask(x, mask)

        reshapingX = torch.reshape(x, [Ax, -1])
        reshapingY = torch.reshape(y, [Ax, -1])
        mask_reshape = torch.reshape(mask, [Ax, -1])

        diff = (reshapingX - reshapingY) ** 2 if self.loss_type == "Loss2" else torch.abs(reshapingX - reshapingY)
        mask_diff = diff * mask_reshape
        sum_diff = torch.sum(mask_diff, axis=-1)
        norm_count = torch.sum(mask_reshape, axis=-1)
        diff_norm = sum_diff / (norm_count + 1.0)

        return torch.mean(diff_norm)

    def set_mask(self, x, mask):
        Ax, Cx, Hx, Wx = x.shape
        Am, Cm, Hm, Wm = mask.shape

        if Cm == 1:
            mask = mask.repeat(1, Cx, 1, 1)

        mask = F.interpolate(mask, scale_factor=Hx / Hm, mode="nearest")

        return mask


class StyleLoss(nn.Module):
    def __init__(self, VGG16_ACTIVATIONS_LIST=[21], normalize=False, distance="Loss2"):
        super(StyleLoss, self).__init__()

        self.vgg16_activations = VGG16_Activations(VGG16_ACTIVATIONS_LIST)
        self.vgg16_activations.eval()

        self.normalize = normalize
        self.distance_metric = distance
        self.custom_loss = CustomLoss(loss_type=distance)

    def get_features(self, model, x):
        return model(x)

    def mask_features(self, x, mask):
        return x * self.custom_loss.set_mask(x, mask) if mask else x

    def gram_matrix(self, x):
        A, C, H, W = x.shape
        x = x.view(A * C, H * W)
        G = torch.mm(x, x.t())
        return G.div(A * H * W * C)

    def calculate_style_loss(self, model, x, x_gen, mask1=None, mask2=None):
        with torch.no_grad():
            activations_x = self.get_features(model, x)
        for layer in activations_x:
            layer.detach_()

        activations_x_gen = self.get_features(model, x_gen)

        style_loss = 0.0
        for layer_x, layer_x_gen in zip(activations_x, activations_x_gen):
            features_x = self.mask_features(layer_x, mask1) if mask1 else layer_x
            features_x_gen = self.mask_features(layer_x_gen, mask2) if mask2 else layer_x_gen

            gram_x = self.gram_matrix(features_x)
            gram_x_gen = self.gram_matrix(features_x_gen)

            style_loss += self.custom_loss(gram_x, gram_x_gen, mask=None, include_bkgd=True)

        style_loss /= len(activations_x)

        return style_loss

    def forward(self, x, x_gen, mask1=None, mask2=None):
        x = x.cuda()
        x_gen = x_gen.cuda()

        A, C, H, W = x.shape
        upsample2d = nn.Upsample(scale_factor=256 / H, mode="bilinear", align_corners=True)

        x = upsample2d(x)
        x_gen = upsample2d(x_gen)

        style_loss = self.calculate_style_loss(self.vgg16_activations, x, x_gen, mask1=mask1, mask2=mask2)
        reconstruction_loss = self.custom_loss(x, x_gen, mask=None, include_bkgd=True)

        return style_loss, reconstruction_loss
