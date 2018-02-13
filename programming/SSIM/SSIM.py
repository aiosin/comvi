from skimage.measure import compare_ssim
from skimage.io import imread
from skimage.transform import resize
from itertools import combinations

from sklearn.cluster import DBSCAN
import sys
import os

fformats = ('.png','.jpeg','.jpg')

def main():
	print("skimage SSIM test")
	print("-----------------")
	pairwise_ssim()
	files = os.listdir(os.curdir)
	files = filter(lambda x: x.endswith(fformats),files)
	DBSCAN(metric=ddistance,algorithm='brute').fit(files)

def ddistance(x,y):
	x = imread(x)
	y = imread(y)
	x = resize(x,(100,100))
	y = resize(y,(100,100))
	return compare_ssim(x,y,multichannel=True)


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
		f.write(sims)
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