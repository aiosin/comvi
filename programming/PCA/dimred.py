from skimage.io import imread
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.exposure import histogram
from  scipy.stats import skew, kurtosis, entropy, energy_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
import os
from pprint import pprint
_fformats = tuple(['.jpg', '.png', ',jpeg', '.tga'])

def readimages(path=None):
    #cannot create empty array with numpy
    #workaround 
    imdata = []
    i=0
    if(path is None):
       	files = os.listdir("/home/zython/comvi/programming/SSIM/jpg")
       	files = list(filter(lambda x: x.endswith(_fformats),files))
        for file in files:
            #calculating features to create feature vector
            try:
                image = imread(file)
            except FileNotFoundError as e:
                continue
            i+=1
            image = resize(image,(100,100))
            image_flat = imread(file,flatten=True)
            image_flat = resize(image_flat,(100,100))

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