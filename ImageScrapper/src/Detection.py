# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import cv2
import torch

from Model import Model


class detection_model(Model):
    
    def __init__(self):
        # download model
        self.model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

        # load model
        self.model = YOLO(self.model_path, task="detect")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = self.model.to(device)
        
        
    def inference(self, imgs, confidence = 0.5):    
        output = self.model(imgs, verbose=False)[0]    
        results = Detections.from_ultralytics(output)
        
        cleaned_boxes = []
        for i in range(len(results.xyxy)):
            if results.confidence[i] > confidence:
                xyxy = results.xyxy[i]
                cleaned_boxes.append((
                    int(xyxy[0]), # x
                    int(xyxy[1]), # y
                    int(xyxy[2]-xyxy[0]), # w
                    int(xyxy[3]-xyxy[1]), # h
                ))
        
        return cleaned_boxes

if __name__ == "__main__":

    model = detection_model()
    
    
    print(model.inference(cv2.imread("images/damat.png")))