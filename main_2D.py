#! /Users/swilson/Documents/Grad_Folder2018/young_lab_rotation/Project-Documents/mypython/  # local venv

######!/lab/solexa_young/scratch/jon_henninger/tools/venv/bin/python  # server venv

import matplotlib
# matplotlib.use('Agg')
matplotlib.use('tkagg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from pprint import pprint

import methods_2D

def main():
    # user input
    input_params = SimpleNamespace()
    
    # Make adjustments here:
    input_params.parent_path = '/Users/swilson/Documents/Grad_Folder2018/young_lab_rotation/Fiji-Images/test_data_2D'
    input_params.output_path = '/Users/swilson/Documents/Grad_Folder2018/young_lab_rotation/Fiji-Images/output_2D'
    file_extension = '.nd'
    separate_rep_imgs = True

    if not os.path.isdir(input_params.output_path):
        os.mkdir(input_params.output_path)

    #parse directory of data and run analysis on each replicate
    data_files = methods_2D.parse_tree(input_params.parent_path, file_extension)
    
    npuncta_list = []
    exp_list = []
    nuclei_list = []    
    for condition, rep_files in data_files.items():        
            excel_output = pd.DataFrame(columns=['sample', 'replicate_id', 'nuclear_id', 'total_nuc_voxels', 'channel', 'mean_in', 'mean_out', 'norm_mean',
                                                'total_in', 'total_out', 'norm_total'])
            replicate_count = 1
            for rep in rep_files:  #REPLICATES
                data = SimpleNamespace()
                base_file = [f for f in rep if file_extension in f][0]
                data.rep_name = base_file.replace(file_extension,'')
                data.condition = condition
                data.rep_files = rep
                
                data = methods_2D.load_images(data, input_params)
                # data = methods.threshold_test(data, input_params)
                data,nuclei = methods_2D.find_nucleus_2D(data, input_params)
                
                
                nuclei_list.append(nuclei)
                nuclei_list.append(nuclei)

                spots = pd.DataFrame(columns=['nuc_id', 'spot_id', 'channel', 'r', 'c', 'z'])
                spot_count = 0
                for channel in data.pro_imgs:
                    ###    First Attempt    ####
                    ## SWH: step 1 - make definition in methods section called subtract_median(img)
                    ## SWH: step 2 - This will subtract background from the image for input into find_blobs(img)
                    #no_backgrnd_img = methods.subtract_median(data.pro_imgs[channel],data,input_params)
                    #projection = methods.max_project(no_backgrnd_img)
                    #print(projection)
                    #methods.find_blobs(projection)
                    #methods.find_blobs(data.pro_imgs[channel])
                    #print(data.pro_imgs[channel])
                    # find puncta here
                    
                    ###    Second Attempt   ###
                    #no_backgrnd_img = methods_2D.subtract_median(data.pro_imgs[channel],data,input_params)
                    no_backgrnd_img = methods_2D.rolling_ball_subtract(data.pro_imgs[channel])
                    blur_img = methods_2D.gaussian_blur(no_backgrnd_img)
                    npuncta = methods_2D.threshold_puncta(blur_img,data,input_params,channel)
                    npuncta_list.append(npuncta)
                    exp = data.rep_name +"_"+channel
                    exp_list.append(exp)
                    '''
                    steps to consider:
                    
                    1. background subtraction:
                        median filter and subtract
                        OR rolling ball algorithm
                        
                    2. gaussian blur
                    
                    3. Threshold
                    
                    4. Morphological operations (e.g. opening, what type of structuring element, fill holes)
                    
                    5. Watershedding to account for touching objects
                    
                    6. Detect objects and quantify
                    
                    OR
                    
                    1. background subtraction
                    2. Blob detection
                    
                    '''
                    print('DONE')

                
                f = open("num_puncta_per_channel.csv",'w')
                for exp in exp_list:
                    f.write(exp)
                    f.write("\t")
                f.write("\n")
                for n in nuclei_list:
                    f.write(str(n))
                    f.write("\t")
                f.write("\n")
                for n in npuncta_list:
                    f.write(str(n))
                    f.write("\t")
                    
                    
if __name__ == "__main__":
    main()
    print('--------------------------------------')
    print('Completed at: ', datetime.now())
