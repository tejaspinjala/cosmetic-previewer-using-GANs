import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import PIL
import torchvision
from PIL import Image
from utils.data_utils import convert_npy_code
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from losses.align_loss import AlignLossBuilder
import torch.nn.functional as F
from utils.data_utils import load_FS_latent
from utils.seg_utils import save_vis_mask
from utils.model_utils import download_weight
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import dilate_erosion_mask_tensor

from models.clip_model import clip_model

toPIL = torchvision.transforms.ToPILImage()

NOSE_INDEX = 2
FACE_INDEX = 1

class Alignment(nn.Module):

    def __init__(self, opts, net=None):
        super(Alignment, self).__init__()
        self.opts = opts
        if not net:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.load_segmentation_network()
        self.load_downsampling()
        self.setup_align_loss_builder()

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def load_downsampling(self):

        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def setup_align_loss_builder(self):
        self.loss_builder = AlignLossBuilder(self.opts)

    def preprocess_img(self, img_path):
        im = torchvision.transforms.ToTensor()(Image.open(img_path))[:3].unsqueeze(0).to(self.opts.device)
        im = (self.downsample(im).clamp(0, 1) - seg_mean) / seg_std
        return im

    def setup_align_optimizer(self, latent_path=None):
        if latent_path:
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path))).to(self.opts.device).requires_grad_(True)
        else:
            latent_W = self.net.latent_avg.reshape(1, 1, 512).repeat(1, 18, 1).clone().detach().to(self.opts.device).requires_grad_(True)

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_align = opt_dict[self.opts.opt_name]([latent_W], lr=self.opts.learning_rate)

        return optimizer_align, latent_W

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
    
    def create_down_seg(self, latent_in):
        gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2

        # get hair mask of synthesized image
        im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(im)
        return down_seg, gen_im

    def dilate_erosion(self, free_mask, device, dilate_erosion=5):
        free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
        free_mask_D, free_mask_E = cuda_unsqueeze(dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
        return free_mask_D, free_mask_E

    def align_images(self, img_path, target_mask_path, save_name="set_the_save_name", sign='realistic', align_more_region=False, smooth=5,
                     save_intermediate=True):

        device = self.opts.device
        output_dir = self.opts.output_dir
        
        im_name = os.path.splitext(os.path.basename(img_path))[0]
        
        target_mask_im = self.preprocess_img(target_mask_path)
        down_seg, _, _ = self.seg(target_mask_im)
        target_mask = torch.argmax(down_seg, dim=1).long()

        im_name = os.path.splitext(os.path.basename(img_path))[0]

        latent_FS_path = os.path.join(output_dir, 'FS', f'{im_name}.npz')

        latent, latent_F = load_FS_latent(latent_FS_path, device)

        latent_W_path = os.path.join(output_dir, 'W+', f'{im_name}.npy')

        optimizer_align, latent_align = self.setup_align_optimizer(latent_W_path)

        pbar = tqdm(range(self.opts.align_steps1), desc='Align Step 1', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align[:, :6, :], latent[:, 6:, :]], dim=1)
            down_seg, _ = self.create_down_seg(latent_in)

            loss_dict = {}
            ##### Cross Entropy Loss
            ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)
            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            # best_summary = f'BEST ({j+1}) | ' + ' | '.join(
            #     [f'{x}: {y:.4f}' for x, y in loss_dict.items()])

            #### TODO not finished

            loss.backward()
            optimizer_align.step()

        """
        if self.opts.nose_shape != -1:
            Clip = clip_model()

            target_nose = torch.tensor(self.opts.nose_shape).to(self.opts.device)
            
            pbar = tqdm(range(self.opts.nose_align_steps), desc='Nose Align', leave=False)
            for step in pbar:
                optimizer_align.zero_grad()
                latent_in = torch.cat([latent_align[:, :6, :], latent[:, 6:, :]], dim=1)
                clip_prob = self.infernce_clip(Clip, latent_in)
                
                down_seg, _ = self.create_down_seg(latent_in)
                max_seg = down_seg.argmax(1).long()
                new_target = target_mask.clone()

                # replaces nose to whatver nose it just generated
                new_target[target_mask == NOSE_INDEX] = FACE_INDEX
                new_target[max_seg == NOSE_INDEX] = NOSE_INDEX

                down_seg[:, :, new_target[0] == NOSE_INDEX] = 0
                down_seg[:, NOSE_INDEX, new_target[0] == NOSE_INDEX] = 1

                mask_loss = self.loss_builder.cross_entropy_loss(down_seg, new_target)
    
                loss_dict = {}
                ##### Cross Entropy Loss
                ce_loss = self.loss_builder.cross_entropy_loss(clip_prob, target_nose)
                loss_dict["ce_loss"] = ce_loss.item()
                loss = ce_loss + mask_loss

                # print(ce_loss, mask_loss)
    
                loss.backward()
                optimizer_align.step()
        """

        ##############################################

        latent_F_out_new, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                                 start_layer=0, end_layer=3)
        latent_F_out_new = latent_F_out_new.clone().detach()
        
        gen_im, _ = self.net.generator([latent], input_is_latent=True, return_latents=False, start_layer=4,
                                       end_layer=8, layer_in=latent_F_out_new)

        self.save_align_results(save_name, sign, gen_im, latent, latent_F_out_new,
                                save_intermediate=save_intermediate)

    def save_align_results(self, im_name, sign, gen_im, latent_in, latent_F, save_intermediate=True):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Align_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}.npz'.format(im_name))
        if save_intermediate:
            image_path = os.path.join(save_dir, '{}.png'.format(im_name))
            save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())
