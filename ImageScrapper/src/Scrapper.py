import os
from GoogleImageScraper.GoogleImageScraper import GoogleImageScraper
from GoogleImageScraper.patch import webdriver_executable
from Utils import getStop, setStopFile
import Constants


WEBDRIVER_PATH = os.path.normpath(os.path.join(os.getcwd(), "src", "GoogleImageScraper", 'webdriver', webdriver_executable()))

SCRAPE_COUNT = 20
HIDE_BROWSER = False
MIN_RESOLUTION = (500,700)
MAX_RESOLUTION = (2000,2000)



def google_scrape(stop_func=None,query="nose", folder=Constants.RAW_IMAGES_DIR, img_download_callback=None, hide_browser=HIDE_BROWSER):
    image_scraper = GoogleImageScraper(WEBDRIVER_PATH,"","",SCRAPE_COUNT,hide_browser,MIN_RESOLUTION,MAX_RESOLUTION)
    
    image_scraper.headless = hide_browser
    image_scraper.search_key = query
    image_scraper.image_path = folder
    image_scraper.img_download_callback = img_download_callback
    
    image_scraper.find_image_urls(stop_func)
    
def google_scraper(queries, clean_queue=None, lock=None):
    # Creates file if it doesnt exist
    if not os.path.isfile(Constants.FINIHSED_RAW_TXT):
        file = open(Constants.FINIHSED_RAW_TXT, 'w')
        file.close()
    
    def dwn_callback(img_pth):
        if clean_queue is None:
            return
        clean_queue.put(img_pth)
    
    for query in queries:    
        # Create folder if it doesnt exist
        output_folder = os.path.join(Constants.RAW_IMAGES_DIR, query)  
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder) 
            
        google_scrape(stop_func=getStop, query=query, folder=output_folder, hide_browser=True, img_download_callback=dwn_callback)

        if lock is None:
            continue 
        # Records that it finished scraping
        with lock:
            with open(Constants.FINIHSED_RAW_TXT, 'a') as raw_finished_file:
                raw_finished_file.write(f'\n{query}')
                

if __name__ == "__main__":
    setStopFile(False)
    
    output = os.path.join(Constants.RAW_IMAGES_DIR, "test")
    if not os.path.isdir(output):
        os.makedirs(output)
    google_scraper(["nose", "other nose"])