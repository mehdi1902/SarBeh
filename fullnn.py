# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:11:41 2016

@author: mehdi
"""
import os
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure
from skimage.morphology import disk
from sklearn.decomposition import PCA
from sklearn import svm
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.structure import SoftmaxLayer
import pickle
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split



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



n_components = 40
input_dir = '/media/mehdi/New/Works/Projects/Saratan/saratan/photos/Other/'
output_dir = '/media/mehdi/New/Works/Projects/Saratan/saratan/res'

samples = []

#for image in os.listdir(input_dir):
#    if (image.split('.')[-1] in ['jpg', 'JPG', 'png', 'PNG']):
#        samples.append(input_dir+image)
#
#X = []
#Y = []
#for sample in samples:        
#    try:
#        gray_images = np.array([i for i in open_image(sample)])
#    except:
#        print 'can\'t open'
#    #        continue
#    m, n = gray_images[0].shape
#    
#    g_min = np.min(gray_images[1:], axis=0)
#    g_max = np.max(gray_images[1:], axis=0)
#    g_avg = np.average(gray_images[1:], axis=0)
#    
#    selem = disk(5)
#    
#    diff = g_max-g_min
#    equalized = img_as_ubyte(exposure.equalize_hist(diff))
#    equalized = exposure.adjust_gamma(g_max,2)
#
#    X.append(equalized.ravel())
#    print [0,1][sample.split('/')[-1][:2]=='04']
#    Y.append([0,1][sample.split('/')[-1][:2]=='04'])
#
#X = np.array(X)


all_data = ClassificationDataSet(inp=n_components, target=1, nb_classes=2)
#    
    
######
## PCA
#print 'PCA'
#pca = PCA(n_components=n_components)
#X_prim = pca.fit_transform(X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                  



#np.savetxt(output_dir+'/X', X)
#np.savetxt(output_dir+'/X_prim', X_prim)
#np.savetxt(output_dir+'/Y', Y)


#print 'Samples'
#for i in range(len(X_prim)):
#    if Y[i]<>2.:
#        all_data.addSample(X_prim[i], Y[i])
#train_data ,test_data = all_data.splitWithProportion(0.75)
#
#train_data._convertToOneOfMany()
#test_data._convertToOneOfMany()
#
#
#################
## Neural Network
#nn = buildNetwork(train_data.indim, 5, train_data.outdim, outclass=SoftmaxLayer)
#trainer = BackpropTrainer(nn, dataset=train_data, momentum=0.1, verbose=True, weightdecay=0.01)
#
#print 'Learning'
#for i in range(200):
#    trainer.trainEpochs(1)
#    trnresult = percentError( trainer.testOnClassData(),
#                              train_data['class'] )
#    tstresult = percentError( trainer.testOnClassData(
#           dataset=test_data), test_data['class'] )
#
#    print "epoch: %4d" % trainer.totalepochs, \
#          "  train error: %5.2f%%" % trnresult, \
#          "  test error: %5.2f%%" % tstresult



X_train, X_test, Y_train, Y_test = train_test_split(X_prim, Y, test_size=0.2, random_state=0)

clf = svm.SVC()
#clf.fit(X_prim[:int(len(X_prim)*.75)], Y[:int(len(Y)*.75)])
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

gammas = np.logspace(-6, -1, 10)
classifier = GridSearchCV(estimator=clf, cv=cv, param_grid=dict(gamma=gammas))
classifier.fit(X_train, Y_train)







