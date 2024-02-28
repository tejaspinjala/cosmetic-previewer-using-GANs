import sys
sys.path.insert(0,'./src')

import os

from Utils import find_images

import Constants


raw_count = len(find_images(Constants.RAW_IMAGES_DIR))
clean_count = len(find_images(Constants.CLEAN_IMAGES_DIR))
accept_count = len(find_images(Constants.ACCEPTED_IMAGES_DIR))

print("raw body images:", raw_count)
print("clean body images:", clean_count)
print("accepted body images:", accept_count)


for dir in os.listdir(Constants.ACCEPTED_IMAGES_DIR):
    class_dir = os.path.join(Constants.ACCEPTED_IMAGES_DIR, dir)
    print(class_dir, "amount:", len(find_images(class_dir)))