import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import skimage.measure
import mahotas as mh

from skimage.io import imread
from skimage.exposure import histogram
from skimage.transform import resize


from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.mixture import VBGMM
from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap


from scipy.stats import skew, kurtosis, entropy, energy_distance
from scipy.spatial import distance
from pprint import pprint


import timeit
import concurrent.futures



_fformats = tuple(['.jpg', '.png', ',jpeg', '.tga','.bmp'])
#TODO: separate feature extraction and 'data aggretation'
#idea: make separate function 'feature_extraction' and
#another function 'read_image'
#TODO: put vector calculation in its own function
def arr2vec(path=None):
	#cannot create empty array with numpy
	#workaround 
	imdata = []
	files = []
	i=0
	if(path is None):
		#REMINDER change to relative rather than absolute path
		#IDEA:
		#files = os.listdir("/home/zython/comvi/programming/datasets/bmw_subset")
		files = os.getcwd()
		files = os.listdir(files)
		files = list(filter(lambda x: x.endswith(_fformats),files))
		print(files[:10])
	else:
		#REDMINER: cwd is relative to the FOLDER that the python script is launched from
		# this should go without saying but, if you cannot read any images 
		#you need to check if the script is running in the right directory
		path = os.getcwd()
	for file in files:
		#calculating features to create feature vector
		try:
			image = imread(file)
		except FileNotFoundError as e:
			continue
		except Exception as e:
			continue
		i+=1
		#IMPORTANT: 
		image = resize(image,(128,128))
		image_flat = imread(file,flatten=True)
		#
		image_flat = resize(image_flat,(128,128))

		#separating channels using slice
		r_im = image[:,:,0]
		g_im = image[:,:,1]
		b_im = image[:,:,2]

		#moments
		#moments = cv2.HuMoments(image_flat)
		r_mo = skimage.measure.moments(r_im).flatten()
		g_mo = skimage.measure.moments(g_im).flatten()
		b_mo = skimage.measure.moments(b_im).flatten()

		#color histogram features
		r_hist = np.histogram(r_im,bins=8)[0]
		g_hist = np.histogram(g_im,bins=8)[0]
		b_hist = np.histogram(b_im,bins=8)[0]
		r_mean = np.average(r_hist)
		g_mean = np.average(g_hist)
		b_mean = np.average(b_hist)
		r_vx = np.var(r_hist)
		g_vx = np.var(g_hist)
		b_vx = np.var(b_hist)
		# r_skw = skew(r_hist)
		# g_skw = skew(g_hist)
		# b_skw = skew(b_hist)
		r_kurt = kurtosis(r_hist)
		g_kurt = kurtosis(g_hist)
		b_kurt = kurtosis(b_hist)
		r_ent = entropy(r_hist)
		g_ent = entropy(g_hist)
		b_ent = entropy(b_hist)
		r_haralick = textural_features(r_im)
		g_haralick = textural_features(g_im)
		b_haralick = textural_features(b_im)

		#THOUGHT: there *has* to be a better way of doing this.
		fvec =np.array((r_mo,g_mo,b_mo,r_hist,g_hist,b_hist,r_mean,
							g_mean, b_mean,r_vx,g_vx,b_vx,
							#r_skw,g_skw,b_skw,
							r_kurt, g_kurt,b_kurt,r_ent,g_ent,b_ent,
							r_haralick,g_haralick,b_haralick,)).ravel()
		fvec = np.reshape(fvec,-1)
		fvec = np.hstack(fvec)
		imdata.append(fvec)
	return imdata

#little lambda for 'tuplifying' of numpy arrays if needed
tuplify_array = lambda x: tuple(map(tuple,x))

#routing for one image
#for hopefully parallelising the image vector routine
def im2vec(file):
	print(str(file))
	#calculating features to create feature vector
	try:
		image = imread(file)
	except FileNotFoundError as e:
		return
	#except Exception as e:
	#	return
	#IMPORTANT: 
	image = resize(image,(128,128))
	#TODO: find out why those two lines exist
	image_flat = imread(file ,flatten=True)
	image_flat = resize(image_flat,(128,128))

	#separating channels using slice
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

	#THOUGHT: there *has* to be a better way of doing this.
	fvec =np.array((r_mo,g_mo,b_mo,
						 r_haralick,g_haralick,b_haralick,
						 r_hist,g_hist,b_hist,
						 r_mean, g_mean, b_mean,
						 r_vx,g_vx,b_vx,
						 r_skw,g_skw, b_skw,
						 r_kurt, g_kurt,b_kurt,
						 r_ent,g_ent,b_ent,
						)).ravel()
	fvec = np.reshape(fvec,-1)
	fvec = np.hstack(fvec)
	#return a tuple of the file and the image vector so we can reconstruct file vector relationship in the result array
	#TODO:TEST WITH fev
	return (fvec,file)


def simpleim2vec(file):
	try:
		image = imread(file,flatten=True)
	except Exception as e:
		return
	image = resize(image,(128,128))
	#TODO: find out why those two lines exist
	image_flat = imread(file ,flatten=True)
	image_flat = resize(image_flat,(128,128))
	r_im = image[:,:,0]
	g_im = image[:,:,1]
	b_im = image[:,:,2]
	r_mo = skimage.measure.moments(r_im).flatten()
	g_mo = skimage.measure.moments(g_im).flatten()
	b_mo = skimage.measure.moments(b_im).flatten()
	fvec = np.array((r_mo,g_mo,b_mo	)).ravel()
	fvec = np.reshape(fvec,-1)
	fvec = np.hstack(fvec)
	#see im2vec for documentation
	return (fvec,file)
	
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

def dokPCA(arr):
	scaler =  StandardScaler()
	scaler.fit(arr)
	arr =scaler.transform(arr)
	pca =KernelPCA(n_components=2,kernel='rbf')
	X = pca.fit_transform(arr)
	return X
	
#TODO: implement
def doG(arr):
	scaler =  StandardScaler()
	scaler.fit(arr)
	arr =scaler.transform(arr)
	mix = BayesianGaussianMixture(n_components=2)
	x = mix.fit(arr)
	y = mix.sample()
	return x

def nldimred(arr):
	pass 

def do_tsne(arr):
	scaler =  StandardScaler()
	scaler.fit(arr)
	arr =scaler.transform(arr)
	snek = TSNE(n_components=2)
	X = snek.fit_transform(arr)
	return X
	
def do_3_tsne(arr):
	scaler =  StandardScaler()
	scaler.fit(arr)
	arr =scaler.transform(arr)
	snek = TSNE(n_components=3)
	X = snek.fit_transform(arr)
	return X

def do_isomap(arr):
	scaler =  StandardScaler()
	scaler.fit(arr)
	arr =scaler.transform(arr)
	iso = Isomap(n_components=2)
	x = iso.fit_transform(arr)
	return x
#return textural features for a given image
#called haralick features
#these are the following 13 or 14 features calculated per directions (?):
#(directions are important for glcm - grey level correlcation matrix)
#"Angular Second Moment","Contrast","Correlation","Sum of Squares: Variance",
#"Inverse Difference Moment","Sum Average","Sum Variance","Sum Entropy",
#"Entropy","Difference Variance","Difference Entropy",
# "Information Measure of Correlation 1",
# "Information Measure of Correlation 2",n Coefficient"
def textural_features(im):
	#TODO: horrible one liner, not readable => expand
	#returns featurearray of size (4*14,) = (64,)
	return  mh.features.haralick( (im*256).astype(int),compute_14th_feature=True).flatten()
	
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
# computes biggest n regions
def biggest_regions(im, n = 8):
	pass


#compute shape feature array 
#input is binary (greyscale) image matrix (V x € im : x = 0 v x = 1)
def shape_features(im,fourier=False):
	features = []
	#make fourier descriptors for shape
	if fourier:
		#get contours
		contours = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		#TODO: make really sure image is binary 
		#255 * 1 = 255
		#255 * 0 = 0
		im = im *255

		#sorted order is ascending, [-1] is largest
		contour = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

		black = np.zeros((im.shape))

		#dear future me:
		#read as follows, 'put' the contour into a empty matrix, 
		#the contour is the 0th index of the single element array [contour]
		#thiccness of the contour is 1 and the color is 255,255,255
		cv2.drawContours(black,[contour],0,(255,255,255),1)
		#get centroid
		moments = cv2.moments(im)
		x = int(moments["m10"] / moments["m00"])
		y = int(moments["m01"] / moments["m00"] )
		#get distance curve from centroid
		#this is a curve 

		#CENTROID CONTOUR DISTANCE CURVE:
		ccdc = []
		for point in contour:
			ccd = distance.euclidean(point,np.array([x,y]))
			ccdc.append(ccd)
		#TODO: normalize the curve (whatever that means)
		#so F-D is scale-invariant
		#THOUGHT:
		#use softmax function
		#=> maps any vector € R  -> v € R^n sum(v)= 1
		#while keeping relative proportions

		#do fft on said curve
		n = len(ccdc)
		Y = np.fft.fft(y)/n
		Y = np.abs(Y)
		#fill array if fft does not return 16 values in this case
		#we fill the empty gaps with zeros

		Y = Y+(np.arange(16))

		#normalization (scale invariant)
		if(Y.max != 0 ):
			Y = Y / Y.max
		#return the first ~16 coefficients (change if too vague)
		return Y[:16]
		
	#normal image moments since they are scale translation and rotaion invariant
	return cv2.HuMoments(im)
	


def main():

	# start = timeit.default_timer()
	# feature_array = arr2vec()
	# step = timeit.default_timer()
	# feature_array = [im2vec(item) for item in sorted(os.listdir(os.getcwd())) ]
	feature_array= []
	#set max_workers accordingly
	with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
		#yikes, horrible oneliner incomming
		# futures = [executor.submit(im2vec,item) for item in list(filter(lamdba x : x.endswith('.png'), os.listdir(os.getcwd())))]
		files = list(filter(lambda x: x.endswith('.png'), os.listdir(os.getcwd())))
		futures = [executor.submit(im2vec,file) for file in files]
		for future in concurrent.futures.as_completed(futures):
			try:
				feature_array.append(future.result()) 
			except Exception as e:
				print(e)

	#sort the feature array based on the file, so arr2vec and im2vec in parallel should be equal 
	feature_array = sorted(feature_array, key= lambda x: x[1])
	stop  = timeit.default_timer()
	feature_array = [item[0] for item in feature_array if item[0] is not None ]

	coords =doPCA(feature_array)
	X = [i[0] for i in coords]
	Y = [i[1] for i in coords]
	k_coords = dokPCA(feature_array)
	kX = [i[0] for i in k_coords]
	kY = [i[1] for i in k_coords]

	#m_coords = doG(feature_array)
	#mX = [i[0] for i in m_coords]
	#mY = [i[1] for i in m_coords]

	t_coords = do_tsne(feature_array)
	tX = [i[0] for i in t_coords]
	tY = [i[1] for i in t_coords]

	t3_coords = do_3_tsne(feature_array)
	t3X = [i[0] for i in t3_coords]
	t3Y = [i[1] for i in t3_coords]
	t3Z = [i[2] for i in t3_coords]
	scaler =  StandardScaler()
	scaler.fit(feature_array)
	scaled_feature_array =scaler.transform(feature_array)

	shift = MeanShift()
	shift.fit(scaled_feature_array)
	print(shift.labels_)

	i_coords = do_isomap(feature_array)
	iX = [i[0] for i in i_coords]
	iY = [i[1] for i in i_coords]

	plt.scatter(iX,iY)
	plt.title('isomap')
	plt.figure()


	plt.title('normal PCA')
	plt.scatter(X,Y)
	plt.figure()
	plt.scatter(kX,kY)
	plt.title('kernel PCA')
	plt.figure()
	plt.scatter(tX,tY)
	plt.title('tsne')

	fig = plt.figure()
	ax = Axes3D(fig)
	plt.title('tsne')

	ax.scatter(t3X,t3Y,t3Z)
	plt.show()

if __name__ == '__main__':
	main()
