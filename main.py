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

import methods




def main():
    # user input
    input_params = SimpleNamespace()
    
    # Make adjustments here:
    input_params.parent_path = '/Users/swilson/Documents/Grad_Folder2018/young_lab_rotation/Fiji-Images/test_data_3D'
    input_params.output_path = '/Users/swilson/Documents/Grad_Folder2018/young_lab_rotation/Fiji-Images/output_3D'
    file_extension = '.nd'
    separate_rep_imgs = True

    if not os.path.isdir(input_params.output_path):
        os.mkdir(input_params.output_path)

    #parse directory of data and run analysis on each replicate
    data_files = methods.parse_tree(input_params.parent_path, file_extension)
    
    npuncta_list = []
    exp_list = []
    nuclei_list = []
    intensities_hp1 = []
    intensities_med1 = []
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
                
                data = methods.load_images(data, input_params)
                data = methods.find_nucleus_3D(data, input_params)
                
                spots = pd.DataFrame(columns=['nuc_id', 'spot_id', 'channel', 'r', 'c', 'z'])
                spot_count = 0
                
                 ### manders ####
                
                '''
                no_backgrnd_img_488 = methods.subtract_median(data.pro_imgs['ch488'],data,input_params)
                if 'HP1' in data.rep_name:
                    blur_img_488 = methods.gaussian_blur(data.pro_imgs['ch488'])#no_backgrnd_img)                                                                                                                                                                                                                          
                elif 'Med1' in data.rep_name:                                                                                                                
                    blur_img_488 = methods.gaussian_blur(no_backgrnd_img_488)

                no_backgrnd_img_561 = methods.subtract_median(data.pro_imgs['ch561'],data,input_params)
                blur_img_561 = methods.gaussian_blur(no_backgrnd_img_561)

                puncta_labels_488,npuncta_488 = methods.threshold_puncta(blur_img_488,data,input_params,'ch488')
                puncta_labels_561,npunct_561 = methods.threshold_puncta(blur_img_561,data,input_params,'ch561')
                mcc = methods.manders(puncta_labels_488,puncta_labels_561,data.pro_imgs['ch488'],data.pro_imgs['ch561'],data,input_params)
                
                print(data.rep_name)                                                                                                                       
                #print(mcc) 
                '''
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
                    
                
                    print(data.rep_name)
                    print(channel)
                    ###    Second Attempt   ###
                    if channel == 'ch488':
                        #no_backgrnd_img = methods.subtract_median(data.pro_imgs[channel],data,input_params)
                        if 'HP1' in data.rep_name:
                            blur_img = methods.gaussian_blur(data.pro_imgs[channel])#no_backgrnd_img)
                        elif 'Med1' in data.rep_name:
                            blur_img = methods.gaussian_blur(data.pro_imgs[channel])
                            #blur_img = methods.gaussian_blur(data.pro_imgs[channel])
                        puncta_labels,npuncta,puncta,puncta_mask = methods.threshold_puncta(blur_img,data,input_params,channel)
                        npuncta_list.append(npuncta)
                        exp = data.rep_name +"_"+channel
                        exp_list.append(exp)
                        if 'HP1' in data.rep_name:
                            intensities = methods.intensity_at_puncta(puncta_labels,data.pro_imgs['ch561'],npuncta,data,input_params)
                            intensities_hp1.append(intensities)
                        
                        elif 'Med1' in data.rep_name:
                            intensities = methods.intensity_at_puncta(puncta_labels,data.pro_imgs['ch561'],npuncta,data,input_params)
                            intensities_med1.append(intensities)
                        
                        manders_list = []
                        for i in range(1,100):
                            random_puncta_mask = methods.random_regions(npuncta,puncta,data.pro_imgs['ch561'],data,input_params) 
                            manders = methods.manders(random_puncta_mask, data.pro_imgs['ch561'],npuncta, data,input_params)
                            manders_list.append(manders)
                        print(np.mean(manders_list))
                        manders_protein = methods.manders(puncta_mask,data.pro_imgs['ch561'],npuncta,data,input_params)
                        print(manders_protein)
                    print('DONE')

                    ## Colocalization analysis
                    ## Two colors
                    #no_backgrnd_img_488 = methods.subtract_median(data.pro_imgs['ch488'],data,input_params)
                    #no_backgrnd_img_561 = methods.subtract_median(data.pro_imgs['ch561'],data,input_params)
                    #methods.colocalize(no_backgrnd_img_488,no_backgrnd_img_561,data,input_params)#no_backgrnd_img_488,no_backgrnd_img_561,data,input_params)

                
               
                                
                
                npuncta_list.append(npuncta)
                exp = data.rep_name +"_"+channel
                exp_list.append(exp)
                f = open("num_puncta_per_channel_cisplatin.csv",'w')
                for exp in exp_list:
                    f.write(exp)
                    f.write("\t")
                f.write("\n")
                for n in npuncta_list:
                    f.write(str(n))
                    f.write("\t")
    num_bins = 50
    flattened_hp1 = [val for sublist in intensities_hp1 for val in sublist]
    plt.hist(flattened_hp1,num_bins,facecolor='blue',alpha=0.5)
    plt.savefig('histogram_hp1_allimgs'+ '.png', dpi=300)
    plt.close()

    num_bins = 50
    flattened_med1 = [val for sublist in intensities_med1 for val in sublist]
    plt.hist(flattened_med1,num_bins,facecolor='blue',alpha=0.5)
    plt.savefig("histogram_med1_allimgs"+ '.png', dpi=300)
    plt.close()

    
                    
if __name__ == "__main__":
    main()
    print('--------------------------------------')
    print('Completed at: ', datetime.now())
