from skimage.io import imread
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.exposure import histogram
from  scipy.stats import skew, kurtosis, entropy, energy_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_fformats = ['.jpg', '.png', ',jpeg', '.tga']

def readimages(path=None):
    imdata = np.array()
    if(path is None):
       	files = os.listdir(os.curdir)
       	files = list(filter(lambda x: x.endswith(_fformats),files))
        for file in files:
            #calculating features to create feature vector
            image = imread(file)
            image_flat = imread(file,flatten=True)

            #moments
            #TODO: skimage moments only work on flat arrays
            moments = cv2.HuMoments(image_flat)
            #mo = skimage.measure.moments(image)
            print(type(moments))

            #color histogram features
            hist = histogram(image,bins=16)
            mean = np.average(hist)
            vx = np.var(hist)
            skw = skew(hist)
            kurt = kurtosis(hist)
            ent = entropy(hist)
            fvec =np.concatenate(moments,hist,mean,vx,skw,kurt,ent)
            imdata.append(fvec)
        print(imdata.shape)
    return imdata


tuplify_array = lambda x: tuple(map(tuple,x))


def doPCA(arr):
    scaler =  StandardScaler()
    scaler.fit(arr)
    scaler.transform(arr)
    PCA(n_components=2)
    



def main():
    pass

if __name__ == '__main__':
    main()