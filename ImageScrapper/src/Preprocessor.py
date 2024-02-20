import os
import cv2
import time
import math
from Utils import getStop, get_file_path
from sixdrepnet import SixDRepNet
import matplotlib.pyplot as plt
import numpy as np
import torch

from Detection import detection_model
import Constants

SIZE_FACE_MULT = 2
ANGLE_FACE_MULT = 2
MINIMUM_QUALITY = 256

MAX_QUALITY = 1024

# This is mirrored to negative angles as well
# Format
# [ 
#   pitch (up and down),
#   yaw  (side to side),
#   roll (angling side to side)
# ]

ALLOWED_ANGLES = {
    "front": np.array([[0, 13], [0, 13], [0, 13]]),
    "mid": np.array([[0, 35], [12, 25], [0, 35]]),
    "side": np.array([[0, 35], [25, 90], [0, 35]])
    }


def crop_image(img, detect_model :detection_model, res_check=True, visualize=False, bottom_extend=False, max_faces=None):
    # Detect faces
    faces = detect_model.inference(img)

    # No face detected
    if len(faces) == 0:
        print("No faces detected... Rejected")
        return None, None
    
    if max_faces is not None and len(faces) > max_faces:
        print("To many faces detected... Rejected")
        return None, None
    
    bounded_images = []
    angle_images = []
    
    for x, y, w, h in faces:  
        
        mid_x, mid_y = int(x + w/2), int(y + h/2)
        max_size = int(max(w,h) / 2)
        
        total_res = int(max_size * SIZE_FACE_MULT * 2)
        
        # Resolution checker
        if res_check and total_res < MINIMUM_QUALITY:
            print(f"{total_res} Resolution wasnt big enough... Rejected")
            continue
        if visualize:
            print("Resolution: ", max_size * SIZE_FACE_MULT * 2)
            print("Face Resolution: ", max_size * 2)
        
        directoinal_scale = {
            "left": max_size * SIZE_FACE_MULT,
            "right": max_size * SIZE_FACE_MULT,
            "bottom": max_size * SIZE_FACE_MULT,
            "top": max_size * SIZE_FACE_MULT,
        }
        
        bounds =  (
            mid_x - directoinal_scale["left"], # left
            mid_x + directoinal_scale["right"], # right
            mid_y - directoinal_scale["bottom"], # bottom 
            mid_y + directoinal_scale["top"] # top
        )
        
        """
        print("Mid:", (mid_x, mid_y))
        print("shape:", img.shape)
        print("Size:", max_size * SIZE_FACE_MULT)
        """
        
        l, r, b, t = bounds 
        
        l_bounds, r_bounds, b_bounds, t_bounds = (
            max(0,-1*l), 
            max(0,r - img.shape[1]),
            max(0,-1*b),
            max(0,t - img.shape[0]),
            )
        
        if bottom_extend and t_bounds > 0:
            print("Bottom doesnt reach... Rejected")
            continue
            
        
        bounded_image = img[
            max(0, b): min(img.shape[0], t),
            max(0, l): min(img.shape[1], r)
            ]
        
        bounded_image = cv2.copyMakeBorder(
            bounded_image, 
            b_bounds, # bottom
            t_bounds, #top
            l_bounds,  # left
            r_bounds, # right
            cv2.BORDER_CONSTANT #borderType
            )
        
        
        face_center_x, face_center_y = directoinal_scale["left"], directoinal_scale["bottom"]
        if visualize:
            # Draw bounding boxes
            face_start_x, face_start_y  = (int(face_center_x - w /2), int(face_center_y - h /2))
            cv2.rectangle(bounded_image, (face_start_x, face_start_y), (face_start_x + w, face_start_y + h), (255, 0, 0), 2)
           
            
        
        # Cropped Image for angle detection
        angle_size = int(ANGLE_FACE_MULT * max_size)
        angle_image = bounded_image[
            face_center_y - angle_size : face_center_y + angle_size,
            face_center_x - angle_size : face_center_x + angle_size
            ].copy()
        
        bounded_images.append(bounded_image)
        angle_images.append(angle_image)
    
    if len(bounded_images) == 0:
        bounded_images, angle_images = None, None
    
    return bounded_images, angle_images
        

def clean_img(model, path, detect_model: detection_model,
              visualize=False, # printing and displayingh with matplotlib
              res_check=False, # Only accepts a minimum resolution
              save_path=None, # Save path
              bottom_extend=False, # Image must reach the bottom when cropped
              max_faces=None, # Max faces in an image
              reject_greyscale=True, # Rejects greyscale images
              downscale_big=True, # Downscales Images that are to big
              ):
    
    try:
        raw_img = cv2.imread(path)
    except:
        print("Image errored on Cv2 load... Rejected")
        return None, None 
    
    if raw_img is None:
        print("Image failed to load, image is None... Rejected")
        return None, None 
    
    if reject_greyscale:
        greyscale = np.array_equal(raw_img[:,:, 0], raw_img[:,:, 1]) and np.array_equal(raw_img[:,:, 1], raw_img[:,:, 2])
        
        if greyscale:
            print("Images is greyscale... Rejected")
            return None, None
        
    
    imgs, angle_images = crop_image(raw_img, detect_model, visualize=visualize, res_check=res_check, bottom_extend=bottom_extend, max_faces=max_faces)
    # If image isn't valid then return 
    if imgs is None:
        return None, None
    
    imgs_final = []
    directions = []
    
    for i in range(len(imgs)):
        img = imgs[i]
        angle_image = angle_images[i]
        
        pitch, yaw, roll = model.predict(angle_image)
        if visualize:
            print(pitch, yaw, roll)
        
        # Finds valid angle that hte image fits
        valid_direction = None
        
        # Checks the validity of the picture face angle based on the given direction
        for direction in ALLOWED_ANGLES:
            if abs(pitch[0]) < ALLOWED_ANGLES[direction][0, 0] or abs(pitch[0]) > ALLOWED_ANGLES[direction][0, 1]:
                continue
            if abs(yaw[0]) < ALLOWED_ANGLES[direction][1, 0] or abs(yaw[0]) > ALLOWED_ANGLES[direction][1, 1]:
                continue
            if abs(roll[0]) < ALLOWED_ANGLES[direction][2, 0] or abs(roll[0]) > ALLOWED_ANGLES[direction][2, 1]:
                continue
            valid_direction = direction
            break
        
        if downscale_big:
            if img.shape[0] > MAX_QUALITY:
                img = cv2.resize(img, (MAX_QUALITY,MAX_QUALITY))
        
        # Visualize  
        if visualize:
            model.draw_axis(img, yaw, pitch, roll)
            model.draw_axis(angle_image, yaw, pitch, roll)
            
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        
        print("Accepted")
        
        if save_path is not None:
            cv2.imwrite(save_path, img)
        
        imgs_final.append(img)
        directions.append(valid_direction)
        
        
    
    return imgs_final, directions


def clean_raw_image(raw_img,
                    root_clean_dir,
                    root_raw_dir,
                    delete_raw=True,
                    model:SixDRepNet=None,
                    detect_model:detection_model=None,
                    ):
    
    if detect_model is None:
        return None 
    
    cleaned_imgs, directions = clean_img(model, raw_img, detect_model=detect_model, res_check=True)

    # Removes the raw img after it is cleaned    
    if delete_raw:   
        os.remove(raw_img)
        
        
    # If image was not accepted continue
    if cleaned_imgs is None or directions is None:
       return []
    
    cleaned_pths = []
    
    for i in range(len(cleaned_imgs)):
        cleaned = cleaned_imgs[i]
        direction = directions[i]
            
        if direction is None:
            continue
            
        # Gets the relative path to raw root directory
        pth_list = get_file_path(raw_img, root_raw_dir[2:])
        # Gets the images save path
        clean_dir = os.path.join(root_clean_dir, pth_list, direction)

            
        if not os.path.isdir(clean_dir):
            os.makedirs(clean_dir)
            
        # Creates the name
        base_split = os.path.basename(raw_img).split(".")
        new_name = ".".join(base_split[:-1]) + str(i) + "." + base_split[-1]
        final_pth = os.path.join(clean_dir, new_name)
        
        # Save the cleaned image
        cv2.imwrite(final_pth, cleaned)
        
        cleaned_pths.append(final_pth)
    
    return cleaned_pths


    # Preprocessor
def Preprocess(clean_queue, accept_queue, 
               root_clean_dir, root_raw_dir, delete_raw=True):    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    # Weights are automatically downloaded
    model = SixDRepNet(gpu_id=0 if device == "cuda" else -1)
    
    detect_model = detection_model()
    detect_model.model = detect_model.model.to(device)
    
    
    # Loop until errors out from timeout (nothing more to clean)
    while True:
        # End early if wants to exit
        if getStop():
            return
        
        # Gets image and cleans it
        raw_img = clean_queue.get(timeout=60)
        if not os.path.isfile(raw_img):
            continue
        
        final_pths = clean_raw_image(
            raw_img, root_clean_dir, root_raw_dir,
            model=model,
            detect_model=detect_model,
            delete_raw = delete_raw
            )
        
        # Adds to the queue for acceptance
        if accept_queue is not None:
            for final_pth in final_pths:
                accept_queue.put(final_pth)    
  
if __name__ == "__main__":
    device = "cpu"
    
    # Create model    
    # Weights are automatically downloaded
    model = SixDRepNet(gpu_id=0 if device == "cuda" else -1)
        
    detect_model = detection_model()
    detect_model.model = detect_model.model.to(device)
    
    test_dir = "./images/accepted_images/nose/front"
    for img_name in os.listdir(test_dir):
        img_pth = os.path.join(test_dir, img_name)
        clean_img(model, img_pth, detect_model=detect_model, visualize=True, res_check=True, bottom_extend=False)
    
    exit()
    
    body_data_test = "data/raw_images/test/"
    
    for img in os.listdir(body_data_test):
        clean_img(model, os.path.join(body_data_test, img), detect_model=detect_model, body_parser=body_parser, visualize=True, res_check=True, bottom_extend=True)
    
    
    
