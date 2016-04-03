# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:14:02 2016

@author: mehdi
"""
from skimage.io import imread, imshow, imsave
from os import listdir
from random import randint
from sklearn.decomposition import PCA
import numpy as np
from pybrain.utilities import percentError
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


#path = '/media/mehdi/New/Works/Projects/Saratan/saratan/photos/'
cores = []
n_components = 20

#####################
## Open images(cores)
def learn(path, target):
#    for image in listdir(path+'021'):
#        cores.append(imread(path+'021/'+image).ravel())
#    for image in listdir(path+'041'):
#        cores.append(imread(path+'041/'+image).ravel())
    for image in listdir(path):
        cores.append(imread(image).ravel())
    
    Y = np.loadtxt(target)
#    y2 = np.loadtxt(path+'41')
#    Y = np.concatenate((y1, y2))
    
    X = np.array(cores)
    
    all_data = ClassificationDataSet(inp=n_components, target=1, nb_classes=2)
    
    
    ######
    ## PCA
    pca = PCA(n_components=n_components)
    X_prim = pca.fit_transform(X)
    
    #Y = [randint(0,1) for _ in range(len(X_prim))]
    
    
    for i in range(len(X_prim)):
        if Y[i]<>2.:
            all_data.addSample(X_prim[i], Y[i])
    train_data ,test_data = all_data.splitWithProportion(0.75)
    
    train_data._convertToOneOfMany()
    test_data._convertToOneOfMany()
    
    
    #################
    ## Neural Network
    nn = buildNetwork(train_data.indim, 5, train_data.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(nn, dataset=train_data, momentum=0.1, verbose=True, weightdecay=0.01)
    
    for i in range(200):
        trainer.trainEpochs(1)
        trnresult = percentError( trainer.testOnClassData(),
                                  train_data['class'] )
        tstresult = percentError( trainer.testOnClassData(
               dataset=test_data), test_data['class'] )
    
        print "epoch: %4d" % trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult, \
              "  test error: %5.2f%%" % tstresult
    
    
#def test()






