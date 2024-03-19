from PIL import Image
import cv2
import torch
import clip
import os

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode

class clip_model:
    def __init__(self):
        # load model
        self.weights = "pretrained_models/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
        
        self.model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        self.model = self.model.cuda()
        # farl_state = torch.load(self.weights)
        # self.model.load_state_dict(farl_state["state_dict"], strict=False)

        self.nose_types = [
            "face picture with a tiny nose",
            "face picture with a huge nose"
        ]
        """[
            "Fleshy Nose",
            "Turned-Up Nose",
            "Hawk Nose",
            "Greek Nose",
            "Nixon Nose",
            "Roman Nose",
            "Bumpy Nose",
            "Snub Nose",
            "Rounded Nose",
            "Nubian Nose"
        ]"""
        
        # Quality tokenizer
        self.nose_text = clip.tokenize(self.nose_types).cuda()

        self.model.encode_text(self.nose_text)

        self.transforms = Compose([
            Resize(self.model.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.model.visual.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
           
    def inference(self, image):
        # with torch.no_grad():
        self.model.encode_image(image)
        logits_per_image, logits_per_text = self.model(image, self.nose_text)
        nose_probs = logits_per_image.softmax(dim=-1)

        return nose_probs

    def img_preprocess(self, img):
        return self.preprocess(Image.fromarray(img)).unsqueeze(0).cuda()
    
    def torch_prepreocess(self, img):
        return self.transforms(torch.flip(img, dims=[1]))


if __name__ == "__main__":
    model = clip_model()
    
    input_dir = "./input/face"
    for x in os.listdir(input_dir):
        if x == ".ipynb_checkpoints": 
            continue
        img_pth = os.path.join(input_dir, x)
        model_in = model.img_preprocess(cv2.imread(img_pth))

        model_percs = model.inference(model_in).argmax(1).data.cpu().numpy()
        
        print(os.path.basename(img_pth), model.nose_types[model_percs[0]])