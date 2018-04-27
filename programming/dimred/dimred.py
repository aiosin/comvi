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
from sklearn.manifold import TSNE,Isomap,LocallyLinearEmbedding,SpectralEmbedding,MDS


from scipy.stats import skew, kurtosis, entropy #energy_distance
from scipy.spatial import distance
from pprint import pprint


import timeit
import concurrent.futures


#gobal variable of file formats which can be accepted 
_fformats = tuple(['.jpg', '.png', ',jpeg', '.tga','.bmp'])



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
		fvec = im2vec(file)
		fvec = np.reshape(fvec,-1)
		fvec = np.hstack(fvec)
		imdata.append(fvec)
	return imdata

#little lambda for 'tuplifying' of numpy arrays if needed
tuplify_array = lambda x: tuple(map(tuple,x))

#routine for one image
#TODO: write docstring
def im2vec(file):
	#calculating features to create feature vector
	try:
		image = imread(file)
		print('read: '+str(file))
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

	r_avg = np.average(r_im) 
	g_avg = np.average(g_im)
	b_avg = np.average(b_im)



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
						r_avg,g_avg,b_avg,
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

#TODO:implement
def do_dimred(arr,mode=None,components=2):
	scaler = StandardScaler()
	scaler.fit(arr)
	reducer = None
	assert mode is not None
	#beautiful switch case courtesy of cpython
	dimpurple = {
		'PCA' : PCA,
		'kPCA' : KernelPCA,
		'LLE' : LocallyLinearEmbedding,
		'gauss_mix': BayesianGaussianMixture,
		'tsne': TSNE,
		'isomap': Isomap,
		'mds':MDS,
	}
	reducer = dimpurple[mode](n_components=components)
	X = reducer.fit_transform(arr)
	return tuple([item[i] for item in X]for i in range(0,components))

def textural_features(im):
	"""
	def textural_features(im)=> (64,) returns featurearray of size (4*14,)
	return textural features for a given image
	called haralick features
	these are the following 13 or 14 features calculated per directions (?):
	(directions are important for glcm - grey level correlcation matrix)
	"Angular Second Moment","Contrast","Correlation","Sum of Squares: Variance",
	"Inverse Difference Moment","Sum Average","Sum Variance","Sum Entropy",
	"Entropy","Difference Variance","Difference Entropy",
	"Information Measure of Correlation 1",
	"Information Measure of Correlation 2",n Coefficient"
	"""
	#TODO: find out about (im*256).astype(int), afair we dont work with binary images
	return  mh.features.haralick( (im*256).astype(int),compute_14th_feature=True).flatten()
	
#TODO: implement n parameter
#n equals number of regions extracted
def biggest_region(im,n=0):
	assert im.dtype.name.contains('int')
	assert len(im.unique) == 2

	#create label matrix (see connected regions post on SO)
	#connectivity can be 1 or 2
	labels = skimage.measure.label(im,connectivity=1)
	#extract unique labels and count of those labels
	unique, counts = numpy.unique(labels, return_counts=True)
	values = [i for i in zip(unique,counts)]
	#sort for biggest label and save it in value
	value = sorted(values,key=lambda x: x[1],reverse=True)[0]
	#(lazy ?) return biggest region as binary matrix since true 
	return  (labels == value).astype(int)

#return the absolute paths of the files inside a directory
#source: https://stackoverflow.com/questions/9816816/
def absoluteFilePaths(directory):
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			yield os.path.abspath(os.path.join(dirpath, f))


#input is binary (greyscale) image matrix (V x € im : x = 0 v x = 1)
def shape_features(im,fourier=False):
	features = []
	#make fourier descriptors for shape
	if fourier:
		contours = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		im = im *255

		#sorted order is ascending, [-1] is largest
		contour = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

		
		#dear future me:
		#read as follows, 'put' the contour into a empty matrix, 
		#the contour is the 0th index of the single element array [contour]
		#thiccness of the contour is 1 and the color is 255,255,255 (white)
		black = np.zeros((im.shape))
		cv2.drawContours(black,[contour],0,(255,255,255),1)
		#get centroid
		moments = cv2.moments(im)
		x = int(moments["m10"] / moments["m00"])
		y = int(moments["m01"] / moments["m00"] )

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
	#else for scenario that we would have some other sort of shape features
	else:
		pass
	

def asyncim2vec(mode='simple',path=None):
	feature_array= []
	#set max_workers accordingly workers=None equals max amount of cores*2 (or 4?)
	with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
		files = list(filter(lambda x: x.endswith('.png'),absoluteFilePaths(path) ))
		if mode is not None:
			if(mode =='complex'):
				futures = [executor.submit(im2vec,file) for file in files]
			if(mode =='simple'):
				futures = [executor.submit(simpleim2vec,file) for file in files]
		for future in concurrent.futures.as_completed(futures):
			try:
				feature_array.append(future.result()) 
			except Exception as e:
				print(e)
	return feature_array


def main():
	start = timeit.default_timer()
	feature_array = arr2vec()
	step = timeit.default_timer()
	print("arr 2 vec done in: " +str(step - start))
	# feature_array = [im2vec(item) for item in sorted(os.listdir(os.getcwd())) ]
	feature_array= asyncim2vec(mode='complex',path=os.path.abspath(os.getcwd()))

	# bmw_feat = asyncim2vec(mode='complex',path='/home/zython/comvi/programming/datasets/bmw_subset' )
	# flower_feat  = asyncim2vec(mode='complex',path='/home/zython/comvi/programming/datasets/flower_subset/')

	#sort the feature array based on the file, so arr2vec and im2vec in parallel should be equal 
	feature_array = sorted(feature_array, key= lambda x: x[1])
	stop  = timeit.default_timer()
	feature_array = [item[0] for item in feature_array if item[0] is not None ]
	bmw_feat = [item[0] for item in bmw_feat if item[0] is not None ]
	flower_feat = [item[0] for item in flower_feat if item[0] is not None ]

	scaler =  StandardScaler()
	scaler.fit(feature_array)
	scaled_feature_array =scaler.transform(feature_array)

	shift = MeanShift()
	shift.fit(scaled_feature_array)
	print(shift.labels_)

if __name__ == '__main__':
	main()