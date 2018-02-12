from skimage.measure import compare_ssim
from skimage.io import imread

import sys
import os

def main():
    print("skimage SSIM test")
    print("-----------------")


def pairwise_ssim():
    files = os.listdir(os.curdir)
    files = filter(lambda x: x.endswith(".png"),files)
    for im1 in files:
        for im2 in files:
            x = imread(im1)
            y = imread(im2)            
            ssim = compare_ssim(x,y,multichannel=True)
            print("SSIM between: ", str(im1), "and: ", str(im2), ssim)

def single_channel_ssim():
    sim = dict()
    files = os.listdir(os.curdir)
    files = filter(lambda x: x.endswith(".png"),files)
    for i in range(0,3):
        #do for single channel and add results to dict
        for im1 in files:
            for im2 in files:
                x = imread(im1)
                y = imread(im2)
                x = x[:,:,i]  
                y = y[:,:,i]      
                ssim = compare_ssim(x,y,multichannel=True)
                print("SSIM between: ", str(im1), "and: ", str(im2), ssim)

if __name__ == '__main__':
    main()