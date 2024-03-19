import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
from PIL import Image
import torchvision
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.optimizer.ClampOptimizer import ClampOptimizer
from utils.data_utils import convert_npy_code
from losses.blend_loss import BlendLossBuilder
import torch.nn.functional as F
import cv2
from utils.data_utils import load_FS_latent
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import load_image, dilate_erosion_mask_path, dilate_erosion_mask_tensor
from utils.model_utils import download_weight
from models.clip_model import clip_model

toPIL = torchvision.transforms.ToPILImage()
NOSE_INDEX = 2




class Blending(nn.Module):
    def __init__(self, opts, net=None):
        super(Blending, self).__init__()
        self.opts = opts
        if not net:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.load_segmentation_network()
        self.load_downsampling()
        self.setup_blend_loss_builder()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def setup_blend_optimizer(self, latent_path):
        latent_1, latent_F_mixed = load_FS_latent(latent_path, self.opts.device)
        latent_1 = latent_1.requires_grad_()
        latent_F_mixed = latent_F_mixed.requires_grad_()
        
        opt_blend = torch.optim.Adam([latent_1, latent_F_mixed], lr=self.opts.learning_rate)
        # opt_blend = ClampOptimizer(torch.optim.Adam, [interpolation_latent], lr=self.opts.learning_rate)
        
        return opt_blend, latent_1, latent_F_mixed
        """
        latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path))).to(self.opts.device).requires_grad_(True)

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_align = opt_dict[self.opts.opt_name]([latent_W], lr=self.opts.learning_rate)
        
        return optimizer_align, latent_W
        """

    def infernce_clip(self, clip_model, latent_in):
        gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2
        # get hair mask of synthesized image
        im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
        model_in = clip_model.torch_prepreocess(im)
        # print(model_in.shape, model_in.min(), model_in.max())
        output = clip_model.inference(model_in)[0]
        return output
    
    def setup_blend_loss_builder(self):
        self.loss_builder = BlendLossBuilder(self.opts)

    def blend_images(self, img_path, target_mask_path, save_name="set_save_name", sign='realistic'):
        device = self.opts.device
        output_dir = self.opts.output_dir

        im_name = os.path.splitext(os.path.basename(img_path))[0]
        # target_mask_name = os.path.splitext(os.path.basename(target_mask_path))[0]

        I_original = load_image(img_path, downsample=True).to(device).unsqueeze(0)
        # I_target = load_image(target_mask_path, downsample=True).to(device).unsqueeze(0)

        fs_latent_path = os.path.join(output_dir, 'Align_{}'.format(sign), save_name + ".npz")
        
        opt_blend, latent_1, latent_F_mixed = self.setup_blend_optimizer(fs_latent_path)
        # latent_1, latent_F_mixed = load_FS_latent(fs_latent_path, device)

        # latent_original, latent_F_original_mixed = load_FS_latent(os.path.join(output_dir, 'FS', im_name + ".npz"), device)

        with torch.no_grad():
            im = (self.downsample(torchvision.io.read_image(img_path).float().unsqueeze(0).cuda() / 255.0) - seg_mean) / seg_std
            im_down_seg, _, _ = self.seg(im)
            original_mask = torch.argmax(im_down_seg, dim=1)
        
            I_X, _ = self.net.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                               end_layer=8, layer_in=latent_F_mixed)
            I_X_0_1 = (I_X + 1) / 2
            IM = (self.downsample(I_X_0_1) - seg_mean) / seg_std
            down_seg, _, _ = self.seg(IM)
            current_mask = torch.argmax(down_seg, dim=1)

            target_mask = torch.where(current_mask == original_mask, 1, 0).float()
            target_mask = F.interpolate(target_mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()

        if self.opts.nose_shape != -1:
            right, left, bottom, top = self.calculate_nose_box(current_mask[0].detach().cpu().numpy())
            move_amount = int(0.1 * current_mask.shape[1])
            right = max(0, right - move_amount)
            bottom = max(0, bottom - move_amount)
            left = min(original_mask.shape[1], left + move_amount)
            top = min(original_mask.shape[0], top + move_amount)

            target_mask[bottom:top, right:left] = 0
            

            Clip = clip_model()
            target_nose = torch.tensor(self.opts.nose_shape).to(self.opts.device)

        target_mask = target_mask.to(device).unsqueeze(0)
        target_mask = torch.zeros_like(target_mask)
        target_mask[0, 0, 0] = 1
        
        pbar = tqdm(range(self.opts.blend_steps), desc='Blend', leave=False)
        for step in pbar:

            opt_blend.zero_grad()

            #latent_mixed = latent_1 + interpolation_latent.unsqueeze(0) * (latent_original - latent_1)
            # latent_mixed = latent_1
            # print(latent_1)
            
            I_G, _ = self.net.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                               end_layer=8, layer_in=latent_F_mixed)
            # I_G_0_1 = (I_G + 1) / 2

            im_dict = {
                'gen_im': self.downsample_256(I_G),
                'im': I_original,
                'mask': target_mask,
            }
            loss, loss_dic = self.loss_builder(**im_dict)

            if self.opts.nose_shape != -1:
                clip_prob = self.infernce_clip(Clip, latent_1)
                
                ce_loss = 0.1 * self.loss_builder.cross_entropy_loss(clip_prob, target_nose)
                print(loss, ce_loss, clip_prob)
                loss += ce_loss
            
            # if self.opts.verbose:
            #     pbar.set_description(
            #         'Blend Loss: {:.3f}, face: {:.3f}, hair: {:.3f}'
            #             .format(loss, loss_dic['face'], loss_dic['hair']))
            loss.backward()
            opt_blend.step()

        ############## Load F code from  '{}_{}.npz'.format(im_name_1, im_name_2)
        # _, latent_F_mixed = load_FS_latent(os.path.join(output_dir, 'Align_{}'.format(sign), save_name + ".npz"), device)
        I_G, _ = self.net.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                           end_layer=8, layer_in=latent_F_mixed)

        self.save_blend_results(save_name, sign, I_G, latent_1, latent_F_mixed)

    # Calculates the bounding box information from the semantic label
    def calculate_nose_box(self, mask):
        top, bottom, right, left = 0, 0, 0, 0
        
        # Gets positions where the data is not 0
        contains_pos = np.argwhere(mask == NOSE_INDEX)
        
        # Handles empty case
        if contains_pos.shape[0] == 0:
            return right, left, bottom, top
        
        # min index where element is not zero
        mins = np.min(contains_pos, axis=0)
    
        bottom = mins[0]
        right = mins[1]
        
        # Max index where element is not zero
        maxs = np.max(contains_pos, axis=0)
        top = maxs[0]
        left = maxs[1]
        
        return right, left, bottom, top

    def save_blend_results(self, save_name, sign,  gen_im, latent_in, latent_F):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Blend_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}.npz'.format(save_name))
        image_path = os.path.join(save_dir, '{}.png'.format(save_name))
        output_image_path = os.path.join(self.opts.output_dir, '{}_{}.png'.format(save_name, sign))

        save_im.save(image_path)
        save_im.save(output_image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())