# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:21:14 2016

@author: mehdi
"""

#from segmentation import segmentation
from skimage.io import imread, imshow, imsave
import segmentation
import os
import numpy as np


input_dir = '/media/mehdi/New/Works/Projects/Saratan/saratan/photos/'
output_dir = '/media/mehdi/New/Works/Projects/Saratan/saratan/res'
#cores = segmentation(sample)


samples = []
cores = []
C = []
cnt = 0
for image in os.listdir(input_dir):
    if (image.split('.')[-1] in ['jpg', 'JPG', 'png', 'PNG']):
        samples.append(input_dir+image)
for sample in samples:
#    print sample
    c = segmentation.segmentation(sample)
    print cnt, ': ', len(c)
    cnt += 1
    
#    print len(c)
#    for core in c:
#        print len(core)
#        print '-------------'
#        if np.all(cores[:]==core)==False:
#        print (core in cores)==False
#        if (core in cores)==False:
#            print '###########'
#            cores.append(core)
#        else:
#            print 'duplicated'
    C.append([c])
    cores.extend(c)

#                for i in range(counter, counter+len(cores)):
#uniques = list(np.unique(np.array(cores)))

start = 0
for i in range(len(cores)):
    image_name = '%s/%i.png'%(output_dir, start+i)
    cnt += 1    
    imsave(image_name, cores[i])
#                counter += len(cores)

