import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

import skimage.measure
import mahotas as mh

from skimage.io import imread
from skimage.exposure import histogram
from skimage.transform import resize


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import skew, kurtosis, entropy, energy_distance

from pprint import pprint


_fformats = tuple(['.jpg', '.png', ',jpeg', '.tga','.bmp'])
#TODO: separate feature extraction and 'data aggretation'
#idea: make separate function 'feature_extraction' and
#another function 'read_image'
#
def readimages(path=None):
    #cannot create empty array with numpy
    #workaround 
    imdata = []
    i=0
    if(path is None):
        #REMINDER change to relative rather than absolute path
       	files = os.listdir("/home/zython/comvi/programming/SSIM/jpg")
       	files = list(filter(lambda x: x.endswith(_fformats),files))
        for file in files:
            #calculating features to create feature vector
            try:
                image = imread(file)
            except FileNotFoundError as e:
                continue
            i+=1
            #IMPORTANT:
            image = resize(image,(128,128))
            image_flat = imread(file,flatten=True)
            #
            image_flat = resize(image_flat,(128,128))

            #separating channels using
            r_im = image[:,:,0]
            g_im = image[:,:,1]
            b_im = image[:,:,2]

            #moments
            #moments = cv2.HuMoments(image_flat)
            r_mo = skimage.measure.moments(r_im).flatten()
            g_mo = skimage.measure.moments(g_im).flatten()
            b_mo = skimage.measure.moments(b_im).flatten()

            #color histogram features
            r_hist = np.histogram(r_im,bins=16)[0]
            g_hist = np.histogram(g_im,bins=16)[0]
            b_hist = np.histogram(b_im,bins=16)[0]
            r_mean = np.average(r_hist)
            g_mean = np.average(g_hist)
            b_mean = np.average(b_hist)
            r_vx = np.var(r_hist)
            g_vx = np.var(g_hist)
            b_vx = np.var(b_hist)
            r_skw = skew(r_hist)
            g_skw = skew(g_hist)
            b_skw = skew(b_hist)
            r_kurt = kurtosis(r_hist)
            g_kurt = kurtosis(g_hist)
            b_kurt = kurtosis(b_hist)
            r_ent = entropy(r_hist)
            g_ent = entropy(g_hist)
            b_ent = entropy(b_hist)
            r_haralick = textural_features(r_im)
            g_haralick = textural_features(g_im)
            b_haralick = textural_features(b_im)
            fvec =np.array((r_mo,g_mo,b_mo,r_hist,g_hist,b_hist,r_mean,
                                g_mean, b_mean,r_vx,g_vx,b_vx,r_skw,g_skw,
                                b_skw,r_kurt, g_kurt,b_kurt,r_ent,g_ent,b_ent)).ravel()
            fvec = np.reshape(fvec,-1)
            fvec = np.hstack(fvec)
            imdata.append(fvec)
    print(i)
    return imdata

#little lambda for 'tuplifying' of numpy arrays if needed
tuplify_array = lambda x: tuple(map(tuple,x))

'''
method doPCA: returns 2d coordinates of dimensionality reduction on given data
'''
def doPCA(arr):
    scaler =  StandardScaler()
    scaler.fit(arr)
    arr =scaler.transform(arr)
    pca =PCA(n_components=2)
    X = pca.fit_transform(arr)
    return X
    
    
#return textural features for a given image
#called haralick features
#these are the following 13 or 14 features calculated per directions (?):
#(directions are important for glcm - grey level correlcation matrix)
#"Angular Second Moment","Contrast","Correlation","Sum of Squares: Variance",
#"Inverse Difference Moment","Sum Average","Sum Variance","Sum Entropy",
#"Entropy","Difference Variance","Difference Entropy",
# "Information Measure of Correlation 1",
# "Information Measure of Correlation 2",n Coefficient"
#
def textural_features(im):
    #TODO: horrible one liner, not readable => expand
    #returns featurearray of size (4*14,) = (64,)
    return features = mh.features.haralick( (im*256).astype(int),compute_14th_feature=True).flatten()
    
#TODO: implement
#REMINDER: we're only interested in the biggest one based on which we will 
#again extract shape, density, entropy, and other features
#aswell as detecting specs of color
def biggest_region(im):
    #we want a binary image, so needs type int (any int)
    #and two distinct 
    assert im.dtype.name.contains('int')
    assert len(im.unique) == 2

    #earlier approach, similar to floodfill algorithm
    #create array of arrays which initially hold the coordinates of all
    #"positive" values of the image
    #further assumption is that all "negative" values are 0
    #use: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.nonzero.html
    #coords = zip(*np.nonzero(im))
    #TODO: test with connectivity 1 and 2

    #create label matrix (see connected regions post on SO)
    labels = skimage.measure.label(im,connectivity=1)
    #extract unique labels and count of those labels
    unique, counts = numpy.unique(labels, return_counts=True)
    values = [i for i in zip(unique,counts)]
    #sort for biggest label and save it in value
    value = sorted(values,key=lambda x: x[1],reverse=True)[0]
    #(lazy ?) return biggest region as binary matrix since true 
    return  (labels == value).astype(int)

#TODO: implement
# copmutes biggest n regions
def biggest_regions(im, n = 8):
    pass


#compute shape feature array 
#input is binary (greyscale) image matrix (V x € im : x = 0 ^ x = 1)
def shape_features(im,fourier=False):
	features = []
    #make fourier descriptors for shape
    if fourier:
        #get contours
        contour = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #get centroid
        moments = cv2.moments(im)
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"] )
        #get curve from centroid
        #do fft on said curve
        #return the first ~16 coefficients (change if too vague)
        
	#normal image moments since they are scale translation and rotaion invariant
    return cv2.HuMoments(im)
	


def main():
    feature_array = readimages()
    coords =doPCA(feature_array)
    X = [i[0] for i in coords]
    Y = [i[1] for i in coords]
    print(coords)
    print(X,Y)
    plt.scatter(X,Y)
    plt.show()

if __name__ == '__main__':
    main()