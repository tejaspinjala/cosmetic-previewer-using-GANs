import os

# Hair Styles
QUERIES_FILES = "./queries.txt"

# Faces files
RAW_IMAGES_DIR = "./images/raw_images"
CLEAN_IMAGES_DIR = "./images/clean_images"
ACCEPTED_IMAGES_DIR = "./images/accepted_images"

FINIHSED_RAW_TXT = "./images/finished_raw"

FLICKR_CREDS_FILE = "./flickr_creds"

# Stop file
STOP_FILE  = "stop.json"


# Process counts
GOOGLE_CLEAN_PROCESSES = 1
GOOGLE_SCRAPE_PROCESSES = 6


def make_dirs():
    # Makes hair importart paths
    if not os.path.exists(RAW_IMAGES_DIR):
        os.makedirs(RAW_IMAGES_DIR)
    if not os.path.exists(CLEAN_IMAGES_DIR):
        os.makedirs(CLEAN_IMAGES_DIR)
    if not os.path.exists(ACCEPTED_IMAGES_DIR):
        os.makedirs(ACCEPTED_IMAGES_DIR)
        
        
  