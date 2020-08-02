#Python Package imports

from sklearn import datasets
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Added the following package for calculating median values
import statistics

import csv
import os, fnmatch
import pandas as pd
import shutil
from shutil import *
import subprocess
import numpy as np

%matplotlib inline  

import os
folder = "C:\Python\powercor_assets\\"
os.chdir(folder)

import pandas as pd
#import geopandas
from shapely.geometry import Point

#Read mid file of coordinates

import fiona
from shapely.geometry import shape
import numpy as np

points = fiona.open("C:\Python\powercor_assets\\transformer.pri_sec_location.mid")

geoms = [ shape(xform["geometry"]) for xform in points ]

list_arrays = [ np.array((geom.xy[0][0], geom.xy[1][0])) for geom in geoms ]
list_arrays = [ (geom.xy[0][0], geom.xy[1][0]) for geom in geoms ]


for array in list_arrays:
    print(array[0:2][0] , array[0:2][1])
	
# Enter your api key here 
api_key = ""

#https://pypi.org/project/google-streetview/
#© 2018 Python Software Foundation

import google_streetview.api

for array in list_arrays:
  
    new_path1 = str(folder + str('/') + str(array) + str('_pitch_170'))
    new_path2 = str(folder + str('/') + str(array) + str('_pitch_10'))

    if not os.path.exists(new_path1):
        os.mkdir(new_path1)
        print("Directory " , new_path1 ,  " Created ")
    else:    
        print("Directory " , new_path1 ,  " already exists")
        os.mkdir(new_path1)

    if not os.path.exists(new_path2):
        os.mkdir(new_path2)
        print("Directory " , new_path2 ,  " Created ")
    else:    
        print("Directory " , new_path2 ,  " already exists")
        os.mkdir(new_path2)
        
    params = [{'size': '600x300', 'location': str(array[0:2][1]) + str(',') + str(array[0:2][0]), 'heading': '235', 'pitch': '170', 'key': 'AIzaSyAsK5CXgeQyyumPt3zs2mCeajTsn_oFN3w'}]
    print(params)
    results = google_streetview.api.results(params)
    results.download_links(new_path1)
    
    params = [{'size': '600x300', 'location': str(array[0:2][1]) + str(',') + str(array[0:2][0]), 'heading': '235', 'pitch': '10', 'key': 'AIzaSyAsK5CXgeQyyumPt3zs2mCeajTsn_oFN3w'}]
    print(params)
    results = google_streetview.api.results(params)
    results.download_links(new_path2)
	
#Download google street view images at pitch 10 & pitch 170

new_folder = "C:\Python\powercor_assets\images\\"

for array in list_arrays:
  
    image_path1 = str(folder + str('/') + str(array) + str('_pitch_170') + str('/') + 'gsv_0.jpg')
    image_path2 = str(folder + str('/') + str(array) + str('_pitch_10') + str('/') + 'gsv_0.jpg')

    target_path1 = str(new_folder + str('/') + str(array) + str('_p170') + '.jpg')
    target_path2 = str(new_folder + str('/') + str(array) + str('_p10') + '.jpg')

    if not os.path.exists(image_path1):
        print("Image " , image_path1 ,  " does not exist ")
    else:    
        print("Image " , image_path1 ,  " found")
        shutil.copy(image_path1, target_path1)
        
    if not os.path.exists(image_path2):
        print("Image " , image_path2 ,  " does not exist ")
    else:    
        print("Image " , image_path2 ,  " found")
        shutil.copy(image_path2, target_path2)
		
