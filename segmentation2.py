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
samples = []
cores = []
cf = .9
core_ratio = .07 #around 10% of image is core
eq_thresh = 10
csize = 150

path = '/media/mehdi/New/Works/Projects/Saratan/saratan/photos/'
for image in os.listdir(path):
    samples.append(path+image)

image_number = 0

for sample in samples:
    if not(sample.split('.')[-1] in ['jpg', 'JPG', 'png', 'PNG']):
        continue
    image_number += 1
    gray_images = np.array([i for i in open_image(sample)])
    m, n = gray_images[0].shape
    
    g_min = np.min(gray_images, axis=0)
    g_max = np.max(gray_images, axis=0)
    g_avg = np.average(gray_images, axis=0)
    
    
    image_gray = g_max-g_min
    edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    
    result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
    print 'here '+i
    result.sort(order='accumulator')
    
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    
    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    gray_images[0][cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(edges)
    edges[cy, cx] = (250, 0, 0)
    
    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True,
                                sharey=True,
                                subplot_kw={'adjustable':'box-forced'})

    ax1.set_title('Original picture')
    ax1.imshow(gray_images[0])
    
    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)
    
    plt.show()
    
    
    
#    selem = disk(5)
#    
#    d = g_max-g_min
#    d1 = g_avg-g_min
#    d2 = g_max-g_avg
#    h2 = histogram(d2)
#    
#    cdf = 0
#    t = g_min.shape[0]*g_min.shape[1]*core_ratio
#    
#    for i in range(len(h2)):
#        cdf += h2[i]
#        if cdf > t:
#            maxi = i
#            break
#    
#    BG = d2 < maxi
#    FG = ~BG
#    
#    '''
#    equalize image -> cores will be white or black
#    '''
##    tmp = g_min
#    tmp = g_max-g_min
##    p = np.where(tmp*FG==0, np.average(tmp), tmp*FG)
#    equalized = img_as_ubyte(exposure.equalize_hist(tmp))
#    #imshow(equalized)
#    
#    
#    #equalized = img_as_ubyte(exposure.equalize_hist(g_min))#g_min
#    equalized = exposure.adjust_gamma(g_max,2)
#    ##eq_mask = []
#    #equalized = img_as_ubyte(exposure.equalize_hist(mask))
#    #eq_mask = equalized<eq_thresh
#    
#    
#    '''
#    local otsu and keep 10% top
#    '''
#    radius = 20
#    selem = disk(radius)
#    local_otsu = rank.otsu(equalized, selem)
##    local_otsu = tmp<threshold_otsu(equalized)
#    bg = tmp<=local_otsu
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
#    
#    
#    lbl, num_lbl = ndi.label(core_mask)
#    
#    lbl, num_lbl = ndi.label(g_max-g_min)
#    #cores = []
#    
#    
#    for i in range(1,num_lbl):
#        c = np.where(np.max(lbl==i, axis=0)==True)[0]
#        left = c[0]
#        right = c[-1]
#        
#        c = np.where(np.max(lbl==i, axis=1)==True)[0]
#        up = c[0]
#        down = c[-1]
#    
#        core = np.zeros((csize, csize))
#        h = down-up
#        w = right-left
#        
#        middle_x = min(max((up+down)/2, csize/2),m-csize/2)
#        middle_y = min(max((left+right)/2, csize/2), n-csize/2)
#        
##        core = (core_mask*gray_images[0])[middle_x-csize/2:middle_x+csize/2, middle_y-csize/2:middle_y+csize/2]
#        core = gray_images[0][middle_x-csize/2:middle_x+csize/2, middle_y-csize/2:middle_y+csize/2]
#        core = exposure.adjust_gamma(core,.5)
#        
#        
#        cores.append(core)

    print 'image', image_number

os.system('rm %sres -R'%(path))
os.system('mkdir %sres'%(path))
#os.system('mkdir %sres/021'%(path))
#os.system('mkdir %sres/041'%(path))
for i in range(len(cores)):
    image_name = '%sres/%i.png'%(path, i)
    imsave(image_name, cores[i])
    
    
    
    
    