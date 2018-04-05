#filesystem related
import os
import sys

#pdb fetching/decoding of json-serialized data
import requests as re
import shutil as sh
import json
from concurrent.futures	import ThreadPoolExecutor

#Life lesson for today: never import requests as re again while simultaniously importing 're'
#import re 
#didnt need regex's anyway

#needed for launching megamol
import subprocess
import xml.etree.ElementTree as ET #call home

#if parallel execution will be a priority at some point
#(multiple instances of megamol)
import threading
import multiprocessing

def extractpdb(file):
	pdb = []
	if file is not None:
		if("pdbesearch" in str(file)):
			with open(file,mode='r',encoding='utf8') as f:
				raw = f.read()
				#file is relatively big, this could be optimized
				json_parsed = json.loads(raw)
				pdb = [key['groupValue'] for key in json_parsed['grouped']['pdb_id']['groups']]
	else:
		print("no file detected")
		return None
	return pdb

#stub for all pdb's available 
#in the database
#full list:
#http://www.rcsb.org/pdb/json/getCurrent
#around 140000 items
def fetchall():
	pdb = []
	resp = re.get("http://www.rcsb.org/pdb/json/getCurrent")
	pdb = [key for key in resp.json()['idList']]
	return pdb
	
#use https://www.rcsb.org/pages/download/http
#to download pdb's to prepare the "pipeline"
#for the next step
def fetchpdb(pdblist,path):
	failed = []
	baseurl = 'https://files.rcsb.org/download/'
	#url for testing purposes in-browser:
	#https://files.rcsb.org/download/4hhb.pdb
	#just makin sure we get a list
	if type(pdblist) == type([]):
		for item in pdblist:
			print('fetching pdb:'+str(item))
			with open(os.path.join(path,str(item)+'.pdb'), mode='wb') as f:
				#try except needed at this point to avoid 404's,
				#and various filesytem related issues
				try:
					print('fetching with url:' + str(baseurl+str(item)+'.pdb' ))
					resp = re.get(baseurl+str(item)+'.pdb')
					#writing bytestream exactly how we got it
					#no idea if pdb's are ascii or not
					f.write(resp.content)   
				#if exceptions occur we do nothing and quietly continue
				#with the next pdb file, for debugging purposes we
				#keep track of the failures 
				except Exception as e:
					failed.append(item)
					print(e)
					print('couldnt download item:'+str(item))
				
#generate_project: generates  a image-generation
#project in the given path  with the given pdb
#the pdb will be downloaded into the current directory,
#given of coursethat we have permissions
#consider this the step prior to launching the megamol to generate the image
def generate_project(pdb,outputpath,projectoutput,currpath):
	if pdb is None:
		raise ValueError('arg pdb should be nonzero')
	if outputpath is None:
		raise ValueError('outputpath should be nonzero')
	if projectoutput is None:
		raise ValueError('projectoutput mustntn\'ve´\'ed be zero')
	if currpath is None:
		raise ValueError('projectoutput mustntn\'ve´\'ed be zero')
		
	with open('baseproject.mmprj','r',encoding='utf8') as f:
		tree = ET.parse(f)
		root = tree.getroot()
		#pdb location
		root[0][2][0].set('value',pdb)
		#output/image location
		root[0][3][4].set('value',outputpath)
		tree.write(os.path.join(projectoutput,pdb+'.mmprj'))
	# with open(os.path.join(outputpath,str(pdb)+'.pdb'),'w') as f:
	# 	baseurl = 'https://files.rcsb.org/download/'
	# 	try:
	# 		resp = re.get(baseurl+str(pdb)+'.pdb')
	# 		f.write(resp.content)
	# 	except Exception as e:
	# 		print('failed download:'+str(pdb))
	

#to future self, this function will probably be no longer needed
#since generate_project does everything you need
def generate_maps(pdb_path,mm_exec,mm_inputpath ,mm_outputdir, comvi_outputdir):
	mm_temp = 'tmp.pdb'
	pdb_root = os.getcwd()
	#'preprocessing'
	#path variable should be the FULL ABSOLUTE PATH to the 
	#for every pdb that we have successfully downloaded
	files = os.listdir(pdb_path)
	for file in files:
		if not file.endswith('.pdb'):
			print(file)
	files = list(filter(lambda x: x.endswith('.pdb'), files))
	#we have now our pdb's in one place
	#that we know the location of
	#time to generate maps
	

	#current idea is to have a megamol configuration/statefile
	#that takes a dummy file 
	#we need to know:
	#	1. the location of the ouput directory 
	#      (methinks megamol does not allow explicit directories)
	#	2. the location of the script/executable that invokes
	#	3. where said executable takes the pdb to generate images from

	for file in file:
		try:
			#1. rename image and move image to the desired location
			os.rename(file,mm_temp)
			filepath = os.path.join(pdb_root,file)
			dest_path  = os.path.join(mm_inputpath,mm_temp)
			sh.move(filepath,dest_path)
			#2. invoke the megamol (which will generate the map correpsonding to the image)
			#	2.1 'correctly' wait for megamol to do its thing
			# assuming megamol will output '0' on exit
			# TODO: implement exit(0) in image save-subroutine
			if subprocess.Popen(mm_exec).wait() !=0:
				#process did not terminate peacefully
				#bring out the big guns (sometime in the future)
				pass

			#3. copy image back with the correct
			filename = file+'.png'
			image_output = os.path.join(comvi_outputdir, filename)
			#FIXME: here be dragons: misleading variable name
			#dest is our src here,
			sh.move(dest_path,image_output)
			
		#4. sometimes megamol cannot handle the image and 'crashes'
		#in this case we need to identify the problem (p.wait will never finish)
		#
		except Exception as e:
			print(e)
			continue


#let the printing commence
def main():
	
	#use this currpath if sys path odesnt work 
	#curr_path = os.path.abspath(os.path.realpath(__file__))
	curr_path = sys.path[0]
	
	print(curr_path)
	# get the pdb dataset folder (in case we want to compare multiple sets)
	# eg. one for testing etc.
	#os.pardir is the identifier to go one directory UP
	dataset_root = os.path.abspath(os.path.join(curr_path,os.pardir,'datasets','pdb'))
	print(dataset_root)
	#getting all the filetypes which could qualify for being a pdb dataset
	#maybe create a tuple somewhere
	pdblist = list(filter(lambda x: x.endswith(('.csv','.json')),os.listdir(dataset_root)))
	print(pdblist[:10])
	
	#currently we grab the first one, because there only is one
	pdbset = pdblist[0]
	print(pdbset)
	os.mkdir(os.path.join(curr_path, 'pdb-dataset'))
	pdb_dir = os.path.join(curr_path,'pdb-dataset')
	print(pdb_dir)

	pdb_names = extractpdb(os.path.join(dataset_root,pdbset))
	print(pdb_names[:10])
	fetchpdb(pdb_names,pdb_dir)

	#maybe download failed for a couple of pdb's
	#also getting the full path here for ease of use with the project files
	available_pdbs = [os.path.join(curr_path,item) for item in  os.listdir(pdb_dir)]
	print(available_pdbs[:10])
	#generating images directory and keeping in absolute
	os.mkdir(os.path.join(curr_path, 'Images'))
	imageoutput  = os.path.join(curr_path,'Images')
	print(imageoutput)

	projectoutput = os.mkdir(os.path.join(curr_path,'Projects'))
	print(projectoutput)
	# third generate the maps
	for item in available_pdbs:
		generate_project(item,imageoutput,projectoutput,curr_path)

	print(os.path.abspath(projectoutput))
	print(list(filter(lambda x: x.endswith('.mmprj'),os.path.abspath(projectoutput))[:10]))
	#in case someone or something puts something inappropriate in project folder
	for item in list(filter(lambda x: x.endswith('.mmprj'),os.path.abspath(projectoutput))):
		#yikes
		subprocess.call(os.path.abspath(os.path.join(curr_path,"Maps","bin","x64","Release","Megamol" )), '-p', item, 'view1 inst')

if __name__ == '__main__':
	main()
