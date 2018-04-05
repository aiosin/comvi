i = 0
try:
	import numpy
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import matplotlib.pyplot
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import tensorflow
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import scipy.stats 
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import sklearn
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import skimage.measure
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import mahotas
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import cv2
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import pprint 
except ModuleNotFoundError as e:
	print(e)
	i += 1

try:
	import requests
except ModuleNotFoundError as e:
	print(e)
	i += 1


def main():
	if(i == 0):
		print('all dependencies met')

if __name__ == '__main__':
	main()


