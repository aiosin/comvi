from skimage.measure import compare_ssim
from skimage.io import imread
from skimage.transform import resize
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from functools import partial

from sklearn.cluster import DBSCAN
import sys
import os
import pickle

fformats = ('.png','.jpeg','.jpg')

def main():
	files = os.listdir(os.curdir)
	files = list(filter(lambda x: x.endswith(fformats),files))
	#files = np.array(files).reshape(-1,1)
	print(np.shape(files))
	dist_matrix  = pairwise_distances(np.arange(len(files)).reshape(-1, 1), metric=partial(ddistance, files),n_jobs=-1)
	print(dist_matrix)
	with open('dist_matr.bin',mode='wb') as f:
		pickle.dump(dist_matrix,f)
	db = DBSCAN(eps=0.05,min_samples=2,metric='precomputed',algorithm='brute').fit(dist_matrix)
	print(vars(db))
	labels = db.labels_
	print(db.labels_)
	cnum = len(set(labels)) - (1 if -1 in labels else 0)
	
	
'''
ddistance custom distance metric for dbscan density based clustering algorithm
'''
def ddistance(files,x,y):
	im1 = files[int(x)]
	im2 = files[int(y)]
	print(im1,im2)
	x = imread(im1)
	y = imread(im2)
	x = resize(x,(100,100))
	y = resize(y,(100,100))
	return compare_ssim(x,y,multichannel=True)

'''
pairwise_ssim: calculates pairwise SSIM distances between a set of images and writes the results to file
'''
def pairwise_ssim():
	files = os.listdir(os.curdir)
	files = filter(lambda x: x.endswith(fformats),files)
	sims = []
	for pair in combinations(files,2):
		x = imread(pair[0])
		y = imread(pair[1]) 
		x = resize(x,(100,100))     
		y = resize(y,(100,100))      
		ssim = compare_ssim(x,y,multichannel=True)
		print(pair[0],pair[1],ssim)
		sims.append((pair[0],pair[1],ssim))
	sims = sorted(sims, key=lambda x: x[2])
	print(sims)
	with open("results",mode='w') as f:
		for item in sims:
			f.write(item)
	return sims

def single_channel_ssim():
	files = os.listdir(os.curdir)
	files = filter(lambda x: x.endswith(fformats),files)
	sims = []
	for pair in combinations(files,2):
		x = imread(pair[0])
		y = imread(pair[1])
		tripels = []
		for i in range(0,3):
			ch1 = x[:,:,i] 
			ch2 = y[:,:,i]
			x = resize(x,(100,100))     
			y = resize(y,(100,100))      
			ssim = compare_ssim(ch1,ch2)		
			tripels.append(ssim)
		sims.append((pair),tuple(tripels))
		

if __name__ == '__main__':
	main()