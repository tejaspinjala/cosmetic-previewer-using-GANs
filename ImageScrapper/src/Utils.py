import json
import os
import math
import Constants
import re
import numpy as np

img_types = ['png', 'jpeg', 'jpg', 'webp']

def setStopFile(stop : bool):
    file = open(Constants.STOP_FILE, 'w')
    json.dump({"Stop": stop}, file)
    file.close()
    
def getStop():
    with open(Constants.STOP_FILE, 'r') as file:
        stop = json.load(file)
        return(stop["Stop"])

# Finds all the images in a directory
def find_images(path: str):
    img_list = []
    
    if os.path.isdir(path):
        for sub in os.listdir(path):
            img_list += find_images(os.path.join(path, sub))
    else:
        name = os.path.basename(path)
        file_type = name.split(".")[-1]
        if file_type in img_types:
            img_list.append(path)
        
    return img_list

# Finds all the images in a directory
def delete_empty(path: str):    
    if os.path.isfile(path):
        return
    if os.path.isdir(path):
        pth_list = os.listdir(path)
        if len(pth_list) == 0:
            os.rmdir(path)
            return
        
        for p in pth_list:
            delete_empty(os.path.join(path, p))
        
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

    
    
def split_query_arr(arr, count : int):
    # Reads file if exists
    already_scraped = []
    if os.path.isfile(Constants.FINIHSED_RAW_TXT):
        with open(Constants.FINIHSED_RAW_TXT, 'r') as scrape_file:
            already_scraped = [x.strip() for x in scrape_file.readlines()]
    
    for x in already_scraped:
        if x in arr:
            arr.remove(x)
        
    amount_per = math.ceil(len(arr) / count)
    split_arrs = [[] for _ in range(count)]
    
    # splits the dictionary
    global_i = 0
    for i in range(count):
        for j in range(amount_per):
            if global_i >= len(arr):
                break
            split_arrs[i].append(arr[global_i])
            global_i += 1
    
    return split_arrs

def get_file_path(pth : str, root_dir : str):
    clean_idx = pth.index(root_dir)
    pth_start = clean_idx + len(root_dir) + 1
    pth_list = "/".join(re.split('[/\\\\]', pth[pth_start:])[:-1])
    return pth_list


def get_flickr_creds():
    # Loads Styles lines from txt file
    # Ignoring comments
    
    assert(os.path.isfile(Constants.FLICKR_CREDS_FILE)), "Flickr creds file doesnt exist"
        
    with open(Constants.FLICKR_CREDS_FILE,'r') as file:
        lines = [x.strip() for x in file.readlines()]
    
    assert(len(lines) >= 2), "Flickr creds file doesnt contain either the key or secret"
    
    key = lines[0]
    secret = lines[1]
    
    return key, secret



def vis_seg(pred):
    num_labels = 19

    color = np.array([[0, 0, 0],  ## 0
                      [102, 204, 255],  ## 1
                      [255, 204, 255],  ## 2
                      [255, 255, 153],  ## 3
                      [255, 255, 153],  ## 4
                      [255, 255, 102],  ## 5
                      [51, 255, 51],  ## 6
                      [0, 153, 255],  ## 7
                      [0, 255, 255],  ## 8
                      [0, 0, 255],  ## 9
                      [204, 102, 255],  ## 10
                      [0, 153, 255],  ## 11
                      [0, 255, 153],  ## 12
                      [0, 51, 0],  # 13
                      [102, 153, 255],  ## 14
                      [255, 153, 102],  ## 15
                      [255, 255, 0],  ## 16
                      [255, 0, 255],  ## 17
                      [255, 255, 255],  ## 18
                      ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]
    return rgb