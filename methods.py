from skimage import filters, exposure
from skimage.color import label2rgb
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as nd
from scipy.spatial import distance
from skimage import img_as_ubyte, img_as_float, img_as_uint
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, blob_log, blob_doh, blob_dog
from skimage import io; io.use_plugin('matplotlib')
from skimage.measure import regionprops
from PIL import Image
from scipy.stats import pearsonr
import random
import imageio as io
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import math
from read_roi import read_roi_zip, read_roi_file
from cv2_rolling_ball import subtract_background_rolling_ball



def parse_tree(parent_path, ext='.nd'):

    # input = path to folder that has sub-folders for each experiment/sample. Replicate files are in each subfolder
    # output = dictionary where key is folder/experiment name and values is a list where each element is a replicate with a list of replicate files
    
    #parse directory of data and run analysis on each replicate
    folder_list = os.listdir(parent_path)
    folder_list.sort(reverse=False)
    
    output = {}
    for folder in folder_list:
        if not folder.startswith('.') and os.path.isdir(os.path.join(parent_path, folder)): #SAMPLES/EXPERIMENTS        
            file_list = os.listdir(os.path.join(parent_path, folder))
            base_name_files = [f for f in file_list if ext in f and os.path.isfile(os.path.join(parent_path, folder, f))]
            base_name_files.sort(reverse=False)
            
            replicate_files = []
            for idx, file in enumerate(base_name_files):  #REPLICATES
                sample_name = file.replace(ext,'')
                replicate_files.append([r for r in file_list if sample_name in r and os.path.isfile(os.path.join(parent_path, folder, r))])
      
            output[folder] = replicate_files
        
    
    return output
    
    
def get_file_extension(file_path):
    file_ext = os.path.splitext(file_path)
    
    return file_ext[1]  # because splitext returns a tuple and the extension is the second element


def find_img_channel_name(file_name):
    str_idx = file_name.find('Conf ')  # this is specific to our microscopes file name format
    channel_name = file_name[str_idx + 5 : str_idx + 8]
    channel_name = 'ch' + channel_name

    return channel_name

def rolling_ball_subtract(img):
    z_size = img.shape[0]

    nobackground_img = np.zeros(shape=img.shape)
    background = np.zeros(shape=img.shape)
    for z in range(z_size):
        temp_img = img[z,:,:]
        nobackground_img[z,:,:], background[z,:,:] = subtract_background_rolling_ball(temp_img, 20, light_background = False, use_paraboloid = False, do_presmooth = False)
    
    return nobackground_img

def max_project(img):
    projection = np.max(img, axis=0)
    return projection

def gaussian_blur(img):
    print("gaussian blur  started")
    z_size = img.shape[0]

    gauss_img = np.zeros(shape=img.shape)
    for z in range(z_size):
        temp_img = img[z, :, :]
        gauss_img[z, :, :] = cv2.GaussianBlur(temp_img, (5,5),0)

    gauss_img = img_as_float(gauss_img)
    return gauss_img

def clear_axis_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    
def make_color_image(img, c):
    output = np.zeros(shape=(img.shape[0], img.shape[0], 3))

    if c == 'green':
        output[..., 0] = 0.0  # R
        output[..., 1] = img  # G
        output[..., 2] = 0.0  # B
    elif c == 'magenta':
        output[..., 0] = img  # R
        output[..., 1] = 0.0  # G
        output[..., 2] = img  # B
    elif c == 'cyan':
        output[..., 0] = 0.0  # R
        output[..., 1] = img  # G
        output[..., 2] = img  # B
        
    else:
        print('ERROR: Could not identify color to pseudocolor image in grapher.make_color_image')
        sys.exit(0)

    return output


def find_region_area(r):
    a = ((r[0].stop - r[0].start)) * ((r[1].stop - r[1].start))
    
    return a


def load_images(data, input_params):    
    # NUCLEUS IMAGE
    nuc_img_file = [f for f in data.rep_files if all(['405 DAPI' in f, get_file_extension(f) == '.TIF'])]
    nuc_img_path = os.path.join(input_params.parent_path, data.condition, nuc_img_file[0])
    data.nuc_img = io.volread(nuc_img_path) # image is [z, x, y] array
    
    if data.nuc_img.dtype == 'float32':
        data.nuc_img = data.nuc_img.astype(np.uint16)  # my pipeline to align image stacks keeps values but in 32-bit format. This conflicts with later processing steps so we have to trick it. Only necessary if the image stacks had to be aligned because of drift.
    
    # PROTEIN IMAGES
    pro_img_paths = [os.path.join(input_params.parent_path, data.condition, p) for p in data.rep_files if all(['405 DAPI' not in p, get_file_extension(p) == '.TIF'])]

    data.pro_imgs = {}
    
    for p in pro_img_paths:
        ch_name = find_img_channel_name(p)
        data.pro_imgs[ch_name] = io.volread(p)
        
        if data.pro_imgs[ch_name].dtype == 'float32':
            data.pro_imgs[ch_name] = data.pro_imgs[ch_name].astype(np.uint16) # my pipeline to align image stacks keeps values but in 32-bit format. This conflicts with later processing steps so we have to trick it.
            
    return data


def read_roi(file):
    try:
        if get_file_extension(file) == '.zip':
            roi = read_roi_zip(file)
        elif get_file_extension(file) == '.roi':
            roi = read_roi_zip(file)
        else:
            print(f'Could not read ROI file for {file}')
            sys.exit(0)
            
        x = []
        y = []
        for key, value in roi.items():
            x.append(int(np.mean(value['x'])))
            y.append(int(np.mean(value['y'])))
        
        success = True
        
        return x, y, success
    
    except:
        print(f'Bad ROI file for {file}')    
        success = False
        return None, None, success


def find_max_z(r, c, ch_img, box_size=3):
    z_num = ch_img.shape[0]
    
    z = []
    for slice in range(z_num):
        z.append(np.mean(ch_img[slice, r-box_size:r+box_size, c-box_size:c+box_size]))
    
    '''
    ### Z-find test
    fig, ax = plt.subplots(1,1)
    
    ax.plot(range(len(z)), z, '-ob', linewidth=2)
    plt.show()
    input('Press enter to continue')
    plt.close()
    '''

    return np.argmax(z)

def threshold_puncta(img, data,input_params,channel):
    z_size = img.shape[0]

    float_img = img_as_float(img)
    
    
    # THRESHOLD TEST
    z = int(float_img.shape[0]/2)
    fig, ax = filters.try_all_threshold(float_img[z,:,:],figsize=(10,8),verbose=False)
    plt.savefig(os.path.join(input_params.output_path, data.rep_name +" "+channel+ 'thresh_test.png'), dpi=300)
    plt.close()
    

    if '6h Cis HP1 HCT 002' in data.rep_name:
        threshold = 2125
    elif '12h Cis HP1 HCT 003' in data.rep_name:
        threshold = 2978
    elif '6h Cis HP1 HCT 001' in data.rep_name:
        threshold = 1734
    elif '6h Cis Med1 HCT 001' in data.rep_name:
        threshold = 743
    elif '6h Cis Med1 HCT 002' in data.rep_name:
        threshold = 569
    elif '6h Cis Med1 HCT 003' in data.rep_name:
        threshold = 562
    else:
        threshold = filters.threshold_triangle(float_img)
    

        
    mask = float_img >= threshold
    puncta_mask = float_img >= threshold

    
    fig, ax = plt.subplots(1,1)
    ax.imshow(exposure.equalize_adapthist(mask[z,:,:]), cmap='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(input_params.output_path, data.rep_name +" "+channel+ 'thresh_test_mean.png'), dpi=300)
    plt.close()    
    
    ### WATERSHED TEST                                                                                                                                    
    labels, num_regions_threshold = nd.label(mask)
    
    distance = nd.distance_transform_edt(mask)

    ##custom faster solution for getting markers                                                                                                          
    sure_fg = distance
    sure_fg[sure_fg <= 0.4*distance.max()] = 0
    sure_fg = sure_fg > 0
    sure_fg = nd.morphology.binary_erosion(sure_fg)

    markers, num_regions = nd.label(sure_fg)
    
    print(num_regions_threshold)
    markers_no_nuc = np.where(data.nuc_mask!=False,labels,0)
    puncta = regionprops(markers_no_nuc)
    total_area = []
    for pt in puncta:    
        total_area.append(pt.area)
        


    row, col = optimum_subplots(mask.shape[0])
    fig, ax = plt.subplots(row, col)
    
    img = img.astype(np.uint16) # not sure if this is allowed
    
    ax = ax.flatten()
    
    for idx, a in enumerate(ax):
        if idx < mask.shape[0]:
            labeled_image = label2rgb(markers[idx, :, :], image=exposure.equalize_adapthist(img[idx, :, :]),
                                 alpha=0.3, bg_label=0, bg_color=[0, 0, 0])
            a.imshow(labeled_image)
        clear_axis_ticks(a)

    plt.savefig(os.path.join(input_params.output_path, data.rep_name +" "+channel+ 'watershed_test.png'), dpi=300)
    plt.close()
        
    return labels,num_regions,puncta,puncta_mask

def set_random_centroid(data):
    z = data.nuc_mask.shape[0]
    

    rand_x = random.randint(0,1023)
    rand_y = random.randint(0,1023)
    rand_z = random.randint(0,z-1)

    
    if data.nuc_mask[rand_z,rand_x,rand_y] == False:
        return set_random_centroid(data)

    
    return (rand_z,rand_x,rand_y)
        

def get_differential(centroid, new_centroid, coords,data,z):
    differential = tuple(np.subtract(new_centroid, centroid))
    for coord in coords:
        if int(coord[0]+differential[0]) < 0 or int(coord[0]+differential[0]) >= z or int(coord[1]+differential[1]) < 0 or int(coord[1]+differential[1]) >=1024 or int(coord[2]+differential[2]) < 0 or int(coord[2]+differential[2]) >= 1024:

            new_c = set_random_centroid(data)
            return get_differential(centroid, new_c, coords,data,z)

    
    return differential
            

## randomize puncta within nuclear regions of an image
def random_regions(markers, puncta,img, data,input_params):
    ## img = 
    ## markers = 
    ## puncta = 
    z = data.nuc_mask.shape[0]
    nuc_mask = data.nuc_mask
    new_mask = np.zeros(shape=img.shape)
    for props in puncta:
        c = props.centroid

        coords = props.coords

        new_c = set_random_centroid(data)

        differential  = get_differential(c,new_c,coords,data,z)#tuple(np.subtract(new_c,c))

        i = 1
        
        for coord in coords:
            new_mask[int(coord[0]+differential[0]),int(coord[1]+differential[1]),int(coord[2]+differential[2])] = i
            

    new_mask = new_mask > 0
    z_slice = int(new_mask.shape[0]/2)
    fig, ax = plt.subplots(1,1)
    ax.imshow(exposure.equalize_adapthist(new_mask[z_slice,:,:]), cmap='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(input_params.output_path, data.rep_name +" "+'random.png'), dpi=300)
    plt.close()
    return new_mask

def intensity_at_puncta(labels_488, img,num_puncta_488,data,input_params):
    img = img.astype(np.uint8)
    #mean = np.mean(img)
    #sd = np.std(img)
    #img = (img - mean)/sd
    z_size = labels_488.shape[0]
    labels_488 = np.where(data.nuc_mask!=False,labels_488,0)
    mean_intensities = []
    for i in range(1,num_puncta_488):
        temp_mask = labels_488 == i
        temp_img = img
        temp_img = temp_img[temp_mask]
        
        if temp_img.size:
            #puncta_max = np.max(temp_img)
            puncta_mean = np.mean(temp_img)
            mean_intensities.append(puncta_mean)
        else:
            print("not in nucleus")
    num_bins = 50
    plt.hist(mean_intensities,num_bins,facecolor='blue',alpha=0.5)
    plt.savefig(os.path.join(input_params.output_path, data.rep_name +"histogram_mean"+ '.png'), dpi=300)
    plt.close()
    return mean_intensities

def manders(puncta_mask,img_561, npuncta,data,input_params):

    sum_puncta_intensities = np.sum(img_561[puncta_mask]) 
    sum_nuclear_intensities = np.sum(img_561[data.nuc_mask])
    
    #return sum_puncta_intensities/sum_nuclear_intensities
        
    #labels_488 = np.where(data.nuc_mask!=False,labels_488,0)
    #img_488 = np.where(labels_488!=0,img_488,0)
    
    #labels_561 = np.where(data.nuc_mask!=False,labels_561,0)
    #img_561 = np.where(labels_561!=0,img_561,0)
    
    

    #max_project_488 = max_project(img_488)
    #max_project_561 = max_project(img_561)

    #img_488 = max_project_488.ravel()
    #img_561 = max_project_561.ravel()

    #sum_pixels = []
    #sum_488_sq = []
    #sum_561_sq = []
    '''
    for i in range(len(img_488)):
        temp_sum = img_488[i]*img_561[i]
        sum_pixels.append(temp_sum)
        temp_488_sq = img_488[i]*img_488[i]
        temp_561_sq = img_561[i]*img_561[i]
        sum_488_sq.append(temp_488_sq)
        sum_561_sq.append(temp_561_sq)
            
    sum = np.sum(sum_pixels)
    sum_488 = np.sum(sum_488_sq)
    sum_561 = np.sum(sum_561_sq)
    manders = sum/(math.sqrt(sum_488*sum_561))
    '''
    return float(sum_puncta_intensities/sum_nuclear_intensities) 
    



def threshold_test(data, input_params):
    img = data.nuc_img
    
    z_size = img.shape[0]
    
    med_img = np.zeros(shape=img.shape)
    for z in range(z_size):
        temp_img = img[z, :, :]
        med_img[z, :, :] = cv2.medianBlur(temp_img, ksize=5)
    
    med_img = img_as_float(med_img)
    

    ## THRESHOLD TEST    
    z = int(med_img.shape[0]/2)

    fig, ax = filters.try_all_threshold(med_img[z, :, :], figsize=(10,8), verbose=False)
    # plt.show()
    
    plt.savefig(os.path.join(input_params.output_path, data.rep_name + 'thresh_test.png'), dpi=300)
    plt.close()
    
    # threshold = filters.threshold_otsu(med_img)
    threshold = filters.threshold_triangle(med_img)
    nuc_mask = med_img >= threshold

    for z in range(z_size):
        nuc_mask[z, :, :] = nd.morphology.binary_fill_holes(nuc_mask[z, :, :])
       
    nuc_mask = nd.morphology.binary_opening(nuc_mask) 

    ### WATERSHED TEST
    labels, _ = nd.label(nuc_mask)
    
    distance = nd.distance_transform_edt(nuc_mask)
    
    ##custom faster solution for getting markers
    sure_fg = distance
    sure_fg[sure_fg <= 0.4*distance.max()] = 0
    sure_fg = sure_fg > 0
    sure_fg = nd.morphology.binary_erosion(sure_fg)
    
    markers, num_regions = nd.label(sure_fg)
    row, col = optimum_subplots(nuc_mask.shape[0])
    fig, ax = plt.subplots(row, col)
    
    ax = ax.flatten()
    
    for idx, a in enumerate(ax):
        if idx < nuc_mask.shape[0]:
            labeled_image = label2rgb(markers[idx, :, :], image=exposure.equalize_adapthist(data.nuc_img[idx, :, :]),
                                 alpha=0.3, bg_label=0, bg_color=[0, 0, 0])
            a.imshow(labeled_image)
        clear_axis_ticks(a)
        
    plt.savefig(os.path.join(input_params.output_path, data.rep_name + 'watershed_test.png'), dpi=300)
    plt.close()

    
def find_nucleus_3D(data, input_params):
    
    img = data.nuc_img
    z_size = img.shape[0]

    if z_size > 0:
        med_img = np.zeros(shape=img.shape)
        for z in range(z_size):
            temp_img = img[z, :, :]
            med_img[z, :, :] = cv2.medianBlur(temp_img, ksize=5)
    
    else:
        med_img[:,:] = cv2.medianBlur(img,ksize=5)
    
    med_img = img_as_float(med_img)
    
    # threshold = filters.threshold_otsu(med_img)
    threshold = filters.threshold_triangle(med_img)
    nuc_mask = med_img >= threshold

    for z in range(z_size):
        nuc_mask[z, :, :] = nd.morphology.binary_fill_holes(nuc_mask[z, :, :])
       
    nuc_mask = nd.morphology.binary_opening(nuc_mask)
    
    # do watershed algorithm 
    nuc_label = find_watershed_3D(data, nuc_mask)
    
    if nuc_label is not None:
        nuc_mask = nuc_label >= 1  
    
        # find nuclei objects
        nuc_label, num_regions = nd.label(nuc_mask)
        print(f'Number of regions is {num_regions}')
    
        data.nuc_mask = nuc_mask
        data.nuc_label = nuc_label
    
        nuc_props = regionprops(nuc_label)
        print(nuc_props[0].area)
        print(nuc_props[0].equivalent_diameter)
        print(nuc_props[0].centroid)
        # data.nuc_regions = nd.find_objects(nuc_label)
        
        ## NUCLEAR MONTAGE
        num_of_z = data.nuc_img.shape[0]
    
        row, col = optimum_subplots(num_of_z)
        fig, ax = plt.subplots(row, col)
        ax = ax.flatten()
    
        for idx, a in enumerate(ax):
            if idx < num_of_z:
                labeled_image = label2rgb(data.nuc_label[idx, :, :], image=exposure.equalize_adapthist(data.nuc_img[idx, :, :]),
                                         alpha=0.3, bg_label=0, bg_color=[0, 0, 0])
                
                a.imshow(labeled_image)
                
                for n in np.unique(data.nuc_label[idx, :, :]):
                    if n != 0:
                        temp_x, temp_y = np.where(data.nuc_label[idx, :, :] == n)
                        i = np.random.randint(len(temp_x))
                        a.text(temp_y[i], temp_x[i], str(n),
                               fontsize='6', color='w', horizontalalignment='center', verticalalignment='center')
        
            clear_axis_ticks(a)
    
        plt.tight_layout()
        plt.savefig(os.path.join(input_params.output_path, data.rep_name + '.png'), dpi=300)
        plt.close()
        return data
    else:
        print(f'Error in {data.rep_name}')
        sys.exit(0)


def subtract_median(img, data,input_params):
    print("median subtraction started")
    z_size = img.shape[0]

    med_img = np.zeros(shape=img.shape)
    for z in range(z_size):
        temp_img = img[z, :, :]
        med_img[z, :, :] = cv2.medianBlur(temp_img, ksize=3)
        
    print("median subtraction done")
    final_img = img - med_img
    final_img = final_img.clip(min=0)

    return final_img

def colocalize(img_1, img_2,data,input_params):
    idx = 16 # z slice to use
    max = "max"

    print("let's colocalize")
    img_1_max = max_project(img_1)
    img_2_max = max_project(img_2)
    img_1_max = img_1_max.astype(np.uint8)
    img_2_max = img_2_max.astype(np.uint8)

    x = img_1_max.ravel()
    y = img_2_max.ravel()
    corr,p_value = pearsonr(x,y)
    print(corr)
    print(p_value)
    #plt.hist2d(x,y,bins = 50, cmap=plt.cm.jet)
    #plt.scatter(x,y)
    #plt.colorbar()
    #plt.savefig(os.path.join(input_params.output_path, data.rep_name +" colocalization " + max+'.png'), dpi=300)
    #plt.close()
    
    '''
    threshold_1 = filters.threshold_triangle(img_1)
    threshold_2 = filters.threshold_triangle(img_2)

    print(threshold_1)
    threshold_1 = threshold_1.astype(np.uint8)
    threshold_2 = threshold_2.astype(np.uint8)
    print(threshold_1)
    img_1_idx = img_1[idx,:,:]
    img_2_idx = img_2[idx,:,:]
    print(img_1_idx)
    img_1_idx = img_1_idx.astype(np.uint8)
    img_2_idx = img_2_idx.astype(np.uint8)

    x = img_1_idx.ravel()
    y = img_2_idx.ravel()
    
    
    for i in range(0,len(x)):
        if x[i] < 43:
            x[i] = 0
            y[i] = 0
        elif y[i] < 53:
            y[i] = 0
            x[i] = 0
    
    x_nozero = []
    y_nozero = []
    for i in range(0,len(x)):
        if x[i] != 0:
            x_nozero.append(x[i])
            y_nozero.append(y[i])
        
    corr,p_value = pearsonr(x_nozero,y_nozero)
    print(corr)
    print(p_value)
    plt.hist2d(x_nozero,y_nozero,bins = 50,cmap=plt.cm.jet)
    plt.colorbar()
    plt.savefig(os.path.join(input_params.output_path, data.rep_name +" colocalization " + str(idx)+ '.png'), dpi=300)
    plt.close()
    '''

def find_watershed_3D(data, vol):
    # vol is the binary nuclear mask
    # sure_bg = nd.morphology.binary_dilation(vol, iterations=3)
    
    '''
    ##  EROSION TEST
    fig, ax = plt.subplots(3, 3)
    ax = ax.flatten()
    
    size = 21
#     z = 10
#     c = make_struct_element(size, shape='circle').astype(vol.dtype)
#     print(c)
#     print(c.dtype)
#     e_t = make_struct_element(size, shape='ellipse_tall').astype(vol.dtype)
    e_w = make_struct_element(size, shape='ellipse_wide').astype(vol.dtype)
        
    c_temp1 = nd.morphology.binary_erosion(vol, structure=c, iterations=1)
    c_temp2 = nd.morphology.binary_erosion(vol, structure=c, iterations=3)
    c_temp3 = nd.morphology.binary_erosion(vol, structure=c, iterations=9)
    
    e_t_temp1 = nd.morphology.binary_erosion(vol, structure=e_t, iterations=1)
    e_t_temp2 = nd.morphology.binary_erosion(vol, structure=e_t, iterations=3)
    e_t_temp3 = nd.morphology.binary_erosion(vol, structure=e_t, iterations=9)
    
    e_w_temp1 = nd.morphology.binary_erosion(vol, structure=e_w, iterations=1)
    e_w_temp2 = nd.morphology.binary_erosion(vol, structure=e_w, iterations=3)
    e_w_temp3 = nd.morphology.binary_erosion(vol, structure=e_w, iterations=9)  # money beet
    
    print(f'Image size is {c_temp1.shape}')
    ax[0].imshow(c_temp1[z, :, :], cmap='gray')
    ax[1].imshow(c_temp2[z, :, :], cmap='gray')
    ax[2].imshow(c_temp3[z, :, :], cmap='gray')
    ax[3].imshow(e_t_temp1[z, :, :], cmap='gray')
    ax[4].imshow(e_t_temp2[z, :, :], cmap='gray')
    ax[5].imshow(e_t_temp3[z, :, :], cmap='gray')
    ax[6].imshow(e_w_temp1[z, :, :], cmap='gray')
    ax[7].imshow(e_w_temp2[z, :, :], cmap='gray')
    ax[8].imshow(e_w_temp3[z, :, :], cmap='gray')
    
    for a in ax:
        clear_axis_ticks(a)
    
    plt.tight_layout()    
    plt.show()
    input('Press enter to continue')
    sys.exit(0)
    '''
    

    labels, _ = nd.label(vol)
    
    distance = nd.distance_transform_edt(vol)
    
    ##custom faster solution for getting markers
    sure_fg = distance
    sure_fg[sure_fg <= 0.2*distance.max()] = 0
    sure_fg = sure_fg > 0
    
    struct_size = 19
    e_w = make_struct_element(struct_size, shape='ellipse_wide').astype(vol.dtype)
    sure_fg = nd.morphology.binary_erosion(sure_fg, structure=e_w, iterations=5)
    # sure_fg = nd.morphology.binary_opening(sure_fg, structure=e_w, iterations=15)  # iterations < 1 keeps it going until nothing changes

    # sure_fg = nd.morphology.binary_erosion(sure_fg)
    
    markers, num_regions = nd.label(sure_fg)
    
    '''
    ### WATERSHED TEST
    row, col = optimum_subplots(vol.shape[0])
    fig, ax = plt.subplots(row, col)
    
    ax = ax.flatten()
    
    for idx, a in enumerate(ax):
        if idx < vol.shape[0]:
            labeled_image = label2rgb(markers[idx, :, :], image=exposure.equalize_adapthist(data.nuc_img[idx, :, :]),
                                 alpha=0.3, bg_label=0, bg_color=[0, 0, 0])
            a.imshow(labeled_image)
        # ax[1].imshow(sure_bg[15, :, :], cmap='Blues')
#         ax[2].imshow(data.nuc_img[15, :, :], cmap='gray')
#         ax[3].imshow(vol[15, :, :], cmap='gray')
        clear_axis_ticks(a)
    
    plt.show()
    
    input('Press enter to continue')
    plt.close()
    sys.exit(0)
    '''
    
    # local_maxi = peak_local_max(distance, min_distance=20, indices=False, labels=labels, exclude_border=True)
    # markers = nd.label(local_maxi)[0]

    output = watershed(-distance, markers, mask=vol, watershed_line=True)
    
    
    return output


def optimum_subplots(n):
    row = math.floor(math.sqrt(n))
    col = math.ceil(n/row)
    
    return row, col


def find_blobs(img):
    #@Jon start here: Blobs might work, but should instead start with Ale's coordinates, and just find the z-slice where the mean signal is highest. Then use that to calculate distance within a nucleus
    # input = channel image that corresponds to same volume of an individual nucleus
    print("this is find blobs")
    blobs = blob_dog(img,  max_sigma=10, threshold=0.01)
    print("done with blob_log")
    if len(blobs) > 1:
        print(blobs)
        
        ### Blob test
        blobs[:, 2] = blobs[:, 2] * math.sqrt(3)
        color = 'lime'
        fig, ax = plt.subplots(1,1)
        
        ax.imshow(exposure.equalize_adapthist(img), cmap='gray')
        
        for blob in blobs:
            y, x, r = blob
            
            c = plt.Circle((x,y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)
            ax.set_axis_off()
        
        plt.tight_layout()
        plt.show()
        
        input('Press enter to continue')
        plt.close()
    else:
        print("no blobs")


def make_struct_element(edge_size, shape='circle'):
    
    if edge_size % 2 == 0:
        print('Error: Structuring element must have an odd edge size')
        sys.exit(0)
    else:
        if shape != 'circle':
            edge_size = edge_size + 2
            
        X, Y = np.ogrid[0:edge_size, 0:edge_size]
        center = math.floor(edge_size/2)
        scale_factor = 1./math.sqrt(edge_size)
        
        if shape == 'circle':
            element = ((X-center)**2 + (Y-center)**2 < edge_size).astype(np.uint8)
        elif shape == 'ellipse_tall':
            element = (scale_factor * (X-center)**2 + (Y-center)**2 < edge_size).astype(np.uint8)
        elif shape == 'ellipse_wide':
            element = ((X-center)**2 + scale_factor * (Y-center)**2 < edge_size).astype(np.uint8)
        else:
            print('Error: Could not recognize shape input for making structuring element')
            sys.exit(0)
            
        # for now, we will just make the z 1 unit thick
        element = element[np.newaxis,:, :]
    
    '''
    ## STRUCT ELEMENT TEST
    print(f'Shape is {shape}')
    print(element)           
    print()
    '''
    
    return element


def get_coords(r):
    
    r_coord = r['r'] * 0.0572  # converting to µm
    c_coord = r['c'] * 0.0572
    z_coord = r['z'] * 0.5  # REMEMBER TO MODIFY THIS DEPENDING ON Z-STEP SIZE
    
    
    return (r_coord, c_coord, z_coord)


def make_output_graphs(nuc_label, obj_mask, output_path):
    num_of_proteins = len(data.protein_images)
    fig_h = 3.3  # empirical
    fig_w = 4.23 * num_of_proteins # empirical
    
    fig, ax = plt.subplots(1, 2+num_of_proteins)
    
    # 1st image is nuclear mask
    nuc_under_img = exposure.equalize_adapthist(max_project(data.nucleus_image))
    nuc_labeled_img = label2rgb(nuc_label, image=nuc_under_img,
                                alpha=0.5, bg_label=0, bg_color=[0, 0 ,0])
    ax[0].imshow(nuc_labeled_img)
    ax[0].set_title('nuclear mask', fontsize=10)

    for r_idx, region in enumerate(data.nuclear_regions):
            region_area = find_region_area(region)
            if region_area >= 10000:
                region_center_r = int((region[0].stop + region[0].start)/2)
                region_center_c = int((region[1].stop + region[1].start)/2)
                nuclear_id = data.nuclear_label[region_center_r, region_center_c]
                ax[0].text(region_center_c, region_center_r, str(nuclear_id),
                           fontsize='6', color='w', horizontalalignment='center', verticalalignment='center')
    
    # 2nd image is total object mask
    object_labeled_img = label2rgb(max_project(obj_mask), image=nuc_under_img,
                                  alpha=0.5, bg_label=0, bg_color=[0, 0, 0])
    ax[1].imshow(object_labeled_img)
    ax[1].set_title('object mask', fontsize=10)
    
    # the rest are the protein images with boxes around
    
    for p_idx, img in enumerate(data.protein_images):
        img = img_as_float(img)
        temp_under_img = exposure.equalize_adapthist(max_project(img))
        temp_labeled_img = label2rgb(max_project(obj_mask), image=temp_under_img,
                               alpha=0.25, bg_label=0, bg_color=[0,0,0])
        ax[2+p_idx].imshow(temp_labeled_img)
        ax[2+p_idx].set_title('ch' + str(data.protein_channel_names[p_idx]))    
    
    for a in ax:
        clear_axis_ticks(a)
        
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.1, hspace=0.1)

    plt.savefig(output_path,dpi=300)
    plt.close()
