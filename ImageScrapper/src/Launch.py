import json
import os


try:
    from getch import getch     # Linux
    def kbhit():
        return False
except ImportError:
    from msvcrt import getch, kbhit  
    
import cv2
from multiprocessing import Process, Queue, Lock
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import tkinter
from tkinter import *
from PIL import Image, ImageTk

from Preprocessor import Preprocess
from Utils import setStopFile, getStop, find_images, \
    split_query_arr, get_file_path, delete_empty
from Querys import get_queries
from Scrapper import google_scraper
# from FlickrScrape.flickr_scraper import body_scrape


import Constants

ACCEPT_IMAGE_SIZE = 500

# Style parser
# Scapes, cleans and shows cleaned image
def parse_style(accept_queue : Queue, root_clean_dir : str, root_accepted_dir : str, clean_processes : list[Process]): 
    
    start_time = time.time()
    
    accept_count = 0
    reject_count = 0
    
    
    root = Tk()
    root.geometry(f"{int(ACCEPT_IMAGE_SIZE)}x{ACCEPT_IMAGE_SIZE + 200}") 
    
    def on_closing():
        setStopFile(True)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Make the window jump above all
    root.attributes('-topmost',True)
    
    description_label = tkinter.Label(root, text="Starting", font=('Times 12'))
    description_label.place(x=int(ACCEPT_IMAGE_SIZE / 2), y=ACCEPT_IMAGE_SIZE, anchor="n", width=ACCEPT_IMAGE_SIZE, height=200)
    
    color_image_label = tkinter.Label(root)
    color_image_label.place(x=0, y=0, anchor="nw", width=ACCEPT_IMAGE_SIZE, height=ACCEPT_IMAGE_SIZE)
    
    while not getStop():
        root.update()

        img_pth = None
        if not accept_queue.empty():
            img_pth = accept_queue.get()

        # If there is not image to accept currently
        if img_pth is None:
            # Only continues images are still being cleaned
            clean_active = False
            for p in clean_processes:
                if p.is_alive():
                    clean_active = True
                    break
            if not clean_active:
                break
            
            description_label.configure(text=f"Waiting for cleaned image... H to stop") 
            
            if kbhit():
                key = getch().decode("utf-8")
                if key == "H" or key == "h":
                    setStopFile(True)
                    break
            
            time.sleep(0.05)
            continue
    
        pth_list = get_file_path(img_pth, root_clean_dir[2:])
                           
        # print("Displaying the image")
        
        # Create a photoimage object of the image in the path
        pil_image = Image.open(img_pth).resize((ACCEPT_IMAGE_SIZE, ACCEPT_IMAGE_SIZE),Image.LANCZOS)
        
        img = ImageTk.PhotoImage(pil_image)
        color_image_label.configure(image=img)
        color_image_label.image = img
            
        description_label.configure(text=f"{pth_list} ||| 'A' to Accept ::: 'R' to reject ::: 'H' to end")    
        
        root.update()
        
        # Gets the images save path
        accepted_dir = os.path.join(root_accepted_dir, pth_list)
        if not os.path.isdir(accepted_dir):
            os.makedirs(accepted_dir)
        accepted_pth = os.path.join(accepted_dir, os.path.basename(img_pth))
        
        while not getStop():
            key = None
            if kbhit():
                key = getch().decode("utf-8")
            else:
                root.update()
                time.sleep(0.05)
                continue
            
            # Moves file to accepted
            if key == "A" or key == "a":
                # Write/ Overwrite accepted image
                if os.path.isfile(accepted_pth):
                    os.remove(accepted_pth)
                os.rename(img_pth, accepted_pth)
                accept_count += 1
                print("Accepted a total of", accept_count)
                break             
            # If rejected delete the image
            elif key == "R" or key == "r":
                os.remove(img_pth)
                reject_count += 1
                print("Rejected a total of", reject_count)
                break
            # Forcefully ends the application
            elif key == "H" or key == "h":
                setStopFile(True)
                break
            else:     
                print("Invalid Key, Input valid key")
            
            # Clears key presses
            while kbhit():
                key = getch()
        
        color_image_label.configure(image=None)
        color_image_label.image = None
            
        description_label.configure(text=f"Waiting...")    
        
        root.update()
    
    root.destroy()   

    total_count = accept_count + reject_count
    if total_count != 0:
        accept_perc = (float(accept_count) / float(total_count)) * 100.0    
        time_took_min = (time.time() - start_time) / 60
        
        print(f"Went throught {total_count} images and accepted {round(accept_perc, 2)}% in {time_took_min}")
        
    print("Ending accepting GUI")

# Starts the application
def launch():  
    torch.multiprocessing.set_start_method('spawn')
    
    # makes important directories
    Constants.make_dirs()
    
    # Sets stop to false
    setStopFile(False)
    
    root_raw_dir = Constants.RAW_IMAGES_DIR
    root_clean_dir = Constants.CLEAN_IMAGES_DIR
    root_accepted_dir = Constants.ACCEPTED_IMAGES_DIR


    print("""
          Choose mode
            1. Scrape, Clean and Accept
            2. Only Scrape and Clean
            3. Only Scrape
            4. Only Clean
            5. Only Accept
          """)
    
    chosen = int(input())
    while chosen < 1 or chosen > 5:
        print(f"{chosen} is an invalid choice, pick a valid choice")
        chosen = int(input())
    
    scrape_processes = []
    clean_processes = []
    
    # Instantiates and fills queue with current images
    clean_queue = Queue() 
    raw_images = find_images(root_raw_dir)
    for img in raw_images:
        clean_queue.put(img)
    print(f"Instantiates clean queue with {len(raw_images)} raw images")
    
    # Instantiates and fills queue with current images
    accept_queue = Queue() 
    clean_images = find_images(root_clean_dir)
    for img in clean_images:
        accept_queue.put(img)
    print(f"Instantiates accept queue with {len(clean_images)} clean images")
    
    scrape_done_lock = Lock()       

    
    
    if chosen <= 4:
        splits = None
        scrape_func = None
        scrape_processes_count = 1
        clean_processes_count = 1
        
        splits = split_query_arr(get_queries(), Constants.GOOGLE_SCRAPE_PROCESSES)
        scrape_func = google_scraper
        scrape_processes_count = Constants.GOOGLE_CLEAN_PROCESSES
        clean_processes_count = Constants.GOOGLE_CLEAN_PROCESSES


        should_scrape = chosen <= 2 or chosen == 3
        should_clean = chosen <= 2 or chosen == 4

        if should_scrape:
            for i in range(scrape_processes_count):
                # Scapes and downloads
                scrape_process = Process(target=scrape_func, args=(splits[i], clean_queue, scrape_done_lock))
                scrape_process.start()
                scrape_processes.append(scrape_process)

        if should_clean:
            for _ in range(clean_processes_count):
                # preprocesses
                clean_process = Process(target=Preprocess, args=(clean_queue, accept_queue, root_clean_dir, root_raw_dir))
                clean_process.start()
                clean_processes.append(clean_process) 
            

    if chosen == 1 or chosen == 5:
        parse_style(accept_queue, root_clean_dir, root_accepted_dir, clean_processes)
    elif chosen >= 2 and chosen <= 4:
        while True:
            scraping = False
            # Checks if still scraping
            for scrape_p in scrape_processes:
                if scrape_p.is_alive():
                    scraping = True
                    break
                
            cleaning = False
            # Checks if still scraping
            for clean_p in clean_processes:
                if clean_p.is_alive():
                    cleaning = True
                    break
                
            if not cleaning and not scraping:
                break
            
            print("'H' TO STOP THE SCAPING/CLEANING")
            if kbhit():
                key = getch().decode("utf-8")
                if key == "H" or key == "h":
                    # Sets stop to True
                    setStopFile(True)
                    break
            
            time.sleep(1)
    
    
    # Ensures stop
    setStopFile(True)
    
    print("Set Stop File")
    
    # Waits for processes to end processes
    for p in scrape_processes:
        if p.is_alive():
            p.kill()
            print("Waiting for Scrape thread to end...")
        p.join()
        
    for c in clean_processes:
        if c.is_alive():
            c.kill()
            print("Waiting for Clean thread to end...")
        c.join()
        
    clean_queue.cancel_join_thread()
    accept_queue.cancel_join_thread()
             
    print("Killed Threads")
    
    # Deletes empty folders
    delete_empty(root_raw_dir)
    delete_empty(root_clean_dir)
    delete_empty(root_accepted_dir)
    
    print("ENDING...")
    
if __name__ == "__main__":
    launch()
    
