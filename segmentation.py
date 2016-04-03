# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:43:37 2015

@author: mehdi
"""

import numpy as np
from skimage.io import imread, imshow, imsave
import glob
import os
from skimage.color import rgb2gray
from skimage.filter import sobel, canny, threshold_otsu, rank, roberts, scharr, prewitt
from skimage.feature import peak_local_max
from skimage import img_as_ubyte
from skimage.morphology import disk, watershed, label, erosion, square, diamond 
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
from skimage import data, color
#from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter





def histogram(image):
    hist = [0]*256
    m, n = image.shape
    for i in range(m):
        for j in range(n):
            hist[int(image[i,j])] += 1
    return hist
    
def open_image(address):
    image = imread(address)
    gray_image = rgb2gray(image)
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
#    
#    m,n,_ = image.shape
#
#    gray_image = img_as_ubyte(gray_image)
    
    return img_as_ubyte(gray_image), img_as_ubyte(r), img_as_ubyte(g), img_as_ubyte(b)

# Open all samples


#input_dir = '/media/mehdi/New/Works/Projects/Saratan/saratan/photos/'
#def segmentation(input_dir, output_dir):
  
#    print samples
#    image_number = 0
  
def segmentation(sample):
    samples = []
    cores = []
    images = []
    cf = .9
    entropy_ratio = .5
    core_ratio = .07 #around 10% of image is core
    eq_thresh = 10
    csize = 300 #change it later
#    for sample in samples:
    #    if image_number<5:
    #        image_number += 1
    #        continue
        
#    image_number += 1
    try:
        gray_images = np.array([i for i in open_image(sample)])
    except:
        print 'can\'t open'
#        continue
    m, n = gray_images[0].shape
    
    g_min = np.min(gray_images[1:], axis=0)
    g_max = np.max(gray_images[1:], axis=0)
    g_avg = np.average(gray_images[1:], axis=0)
    g_mean = rank.mean_bilateral(g_max, disk(40))
    images.append(g_max)
    
    
    selem = disk(5)
    
    diff = g_max-g_min
    diff1 = g_avg-g_min
    diff2 = g_max-g_avg
    h2 = histogram(diff2)
    
    '''
    equalize image -> cores are white or black
    '''
    equalized = img_as_ubyte(exposure.equalize_hist(diff))
    
    
    #equalized = img_as_ubyte(exposure.equalize_hist(g_min))#g_min
    equalized = exposure.adjust_gamma(g_max,2)
    ##eq_mask = []
    #equalized = img_as_ubyte(exposure.equalize_hist(mask))
    #eq_mask = equalized<eq_thresh
    
    
    '''
    local otsu
    '''
    radius = 20
    selem = disk(radius)
    local_otsu = rank.otsu(equalized, selem)
#    local_otsu = tmp<threshold_otsu(equalized)
    bg = diff<=local_otsu
    

    ent = rank.entropy(g_max*~bg, disk(35))
    grad = rank.gradient(g_mean, disk(50))
    tmp = ent*grad
    core_mask = tmp>(np.min(tmp)+(np.max(tmp)-np.min(tmp))*entropy_ratio)    
    
#    
#    h = histogram(local_otsu)
#    cdf = 0
#    t = g_min.shape[0]*g_min.shape[1]*core_ratio
#    
#    for i in range(len(h)):
#        cdf += h[i]
#        if cdf > t:
#            maxi = i
#            break
#        
#    core_mask = (local_otsu<maxi)
##    imshow(core_mask)
#    ##cores = np.logical_and(eq_mask, core_mask)
#    ##imshow(eq_mask)
#    #
#    #
    
    
    lbl, num_lbl = ndi.label(core_mask)
    
    
    for i in range(1,num_lbl+1):
        '''
        lbl==0 is background
        '''
        c = np.where(np.max(lbl==i, axis=0)==True)[0]
        left = c[0]
        right = c[-1]
        
        c = np.where(np.max(lbl==i, axis=1)==True)[0]
        up = c[0]
        down = c[-1]

#        '''
#        Don't consider edge cores
#        '''            
#        if left<csize/2 or right>n-csize/2:
#            continue
#        if up<csize/2 or down>m-csize/2:
#            continue
#        
        
    
        core = np.zeros((csize, csize))
        h = down-up
        w = right-left
        
        middle_x = min(max((up+down)/2, csize/2),m-csize/2)
        middle_y = min(max((left+right)/2, csize/2), n-csize/2)
        
#        core = (core_mask*gray_images[0])[middle_x-csize/2:middle_x+csize/2, middle_y-csize/2:middle_y+csize/2]
        core = gray_images[0][middle_x-csize/2:middle_x+csize/2, middle_y-csize/2:middle_y+csize/2]
        core = exposure.adjust_gamma(core,.5)
        
        
        cores.append(core)
    return cores
#    print 'image', image_number
#    
#if __name__=='__main__':
#    os.system('rm %s -R'%(output_dir))
#    os.system('mkdir %s'%(output_dir))
#    #os.system('mkdir %sres/021'%(input_dir))
#    #os.system('mkdir %sres/041'%(input_dir))
#    for i in range(len(cores)):
#        image_name = '%s/%i.png'%(output_dir, i)
#        imsave(image_name, cores[i])
        
        
        
        
        