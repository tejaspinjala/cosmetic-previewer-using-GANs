import os
import time
import numpy as np
import Constants


def get_queries():
    # Loads Styles lines from txt file
    # Ignoring comments
    styles = [] 
    with open(Constants.QUERIES_FILES,'r') as file:
        for x in file.readlines():
            line = x.strip()
            if len(line) < 2 or line[:2] == "--" or line[0] == "#":
                continue
            styles.append(line)
    
    return styles
            
if __name__ == "__main__":
    print(get_queries())