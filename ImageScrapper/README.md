# FS-HairstyleScraping

Automating Scraping and Cleaning of images. It then creates a simple GUI to show to the user for accepting or rejecting cleaned images

## Description of the Process

- Scrapes Images from supplied queries
- Cleans images based on quality, faces found count and face angle
- Shows the cleaned images to the user to accept or reject


## Launching Program

**Directions**

```sh

# cd into scraping dir
cd ImageScrapper

# Create conda env
conda env create -f environment.yml

# activate the environment
conda activate WebScraping
```

**ALL CODE MUST BE RAN FROM THE 'ImageScrapper' DIRECTORY**

- Put all the styles that want to be scraped into the `queries.txt` file
    - These are the queries that will be search in google images
- Run the below command to launch the program
```sh
cd "git_repo_root"/ImageSrapper
python src/Launch.py
``` 

```sh
# How to get class counts
python ./tools/class_counter.py 
```
