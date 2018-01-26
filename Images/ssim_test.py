import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, io
from skimage.measure import compare_ssim as ssim
from skimage.transform import resize

def bitmap_to_3ch(im):
	img = im.copy()
	r = img[:,:,0]
	g = img[:,:,1]
	b = img[:,:,2]
	return r,g,b
	

def main():
	avg = 0
	#get files
	files = list(filter(lambda x: x.endswith(".png") ,os.listdir()))
	results = dict()
	for p1 in files:
		x = resize(io.imread(p1),(400,200))

		if avg != 0:
			print("avg")
			print(avg/(len(files)-1))
		avg=0
		for p2 in files:
			y = resize(io.imread(p2),(400,200))
			z = ssim(x,y,multichannel=True)
			string = str(p2)+" ssim: "+str(p1)
			print(string+str(z))
			results.update({string:z})
			avg+=z

	for k,v in sorted(results, key=results.get):
		print(k,v)

if __name__ == '__main__':
	#main()
	files = list(filter(lambda x: x.endswith(".png") ,os.listdir()))
	le_file = files[0]
	le_image = io.imread(le_file)
	r,g,b = bitmap_to_3ch(le_image)
	io.imshow(le_image)
	io.show()
	io.imshow_collection([le_image, r,g,b])
	