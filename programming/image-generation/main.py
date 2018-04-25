"""
Module image generation used to generate moledular maps from a list of pdbs with a precompiled molecularmaps binary.
Currently works only under Windows and only with a CUDA compatible card.
Not tested with wine under linux.
If you have custom lists of pdbs which you want to generate images, you should modify the main.py file.
main.py creates three folders:
- Images (contains the final images)
- Projects (contains the project files from the fetched pdbs.)
- pdb (or some variant of that spelling)(contains the fetched pdbs that were downloaded from the protein database)

Usage:
1. unzip the maps.zip so you have a "Maps" folder in the same directory as your main.py
2. get your pdbs and put them into a csv, json file or any parsable format
3. edit the main to extract the pdb files into a list, there are examples on 
   how to work with csv, json and xml files somewhere in this repository
4. run main.py and wait a couple of hours/days 

Known issues:
- image is partly black (no fix)
- megamol crashes unexpectedly (no fix)
- python script crashes (not happened yet, for fix look what images have been generated and
  delete the project files of the images which have already been generated, the script will start with the first available project in the
  projects folder)
- megamol starts speaking french (no fix, just ignore std_err, this is msms.exe speaking, which has been developed by a frenchman)
- megamol generates illegal pngs (no fix, just sort them out before you start working with the imges)
- megamol 

"""

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

def extractpdb(file,start=3000,count=10000):
	pdb = []
	if file is not None:
		if("pdbesearch" in str(file)):
			with open(file,mode='r',encoding='utf8') as f:
				raw = f.read()
				#file is relatively big, this could be optimized
				json_parsed = json.loads(raw)
				pdb = [key['groupValue'] for key in json_parsed['grouped']['pdb_id']['groups']]
	#execution path if no file is given
	else:
		pdb = fetchall()
		#full dataset is ~140k items big
		pdb = pdb[start:start+count]
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
#used for async task execution
def fandwrite(pdb,path):
	baseurl = 'https://files.rcsb.org/download/'
	with open(os.path.join(path,str(pdb)+'.pdb'), mode='wb') as f:
		try:
			print('fetching with url:' + str(baseurl+str(pdb)+'.pdb' ))
			resp = re.get(baseurl+str(pdb)+'.pdb')
			f.write(resp.content)
			print('sucessfully downloaded pdb:'+str(pdb))
		except Exception as e:
			print('error downloading pdb: '+str(pdb))

#TODO:change to parallel if done testing
def fetchpdb(pdblist,path,parallel=True):
	i = 0
	failed = []
	baseurl = 'https://files.rcsb.org/download/'
	#url for testing purposes in-browser:
	#https://files.rcsb.org/download/4hhb.pdb
	#just makin sure we get a list
	if type(pdblist) == type([]):
		if(parallel):
			#to quote greina: this is the apex of suck and must die:
			#executor cant accept noniterable arguments
			patharr = [path]*len(pdblist)
			#if gigabit available, set max_workers accordingly
			#else you'd just starve out the downloads
			with ThreadPoolExecutor(max_workers=50) as executor:
				executor.map(fandwrite,pdblist,patharr)
			return
		for item in pdblist:
			#for testing
			if (i > 10):
				return
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
					i+=1  
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
		
	with open(os.path.join(currpath,'baseproject.mmprj'),'r',encoding='utf8') as f:
		tree = ET.parse(f)
		root = tree.getroot()
		#pdb location
		#root[0][2][0].set('value',os.path.join(currpath,'pdb-dataset',pdb))
		root[0][2][0].set('value',pdb)
		#output/image location important to include pathset or else the image will look like this:
		#Images5uii.png instead of being in the directory Images\5uii
		root[0][3][4].set('value',outputpath+'\\')
		#explanation for following line: we write the file under the project
		# output folder using only the pdb identifier 
		tree.write(os.path.join(projectoutput,os.path.split(pdb)[1][:-4]+'.mmprj'))
	# with open(os.path.join(outputpath,str(pdb)+'.pdb'),'w') as f:
	# 	baseurl = 'https://files.rcsb.org/download/'
	# 	try:
	# 		resp = re.get(baseurl+str(pdb)+'.pdb')
	# 		f.write(resp.content)
	# 	except Exception as e:
	# 		print('failed download:'+str(pdb))
	



#let the printing commence
def main():
	
	#use this currpath if sys path odesnt work 
	#curr_path = os.path.abspath(os.path.realpath(__file__))
	curr_path = sys.path[0]
	
	# get the pdb dataset folder (in case we want to compare multiple sets)
	# eg. one for testing etc.
	#os.pardir is the identifier to go one directory UP
	dataset_root = os.path.abspath(os.path.join(curr_path,os.pardir,'datasets','pdb'))
	#getting all the filetypes which could qualify for being a pdb dataset
	#maybe create a tuple somewhere
	pdblist = list(filter(lambda x: x.endswith(('.csv','.json')),os.listdir(dataset_root)))
	
	#currently we grab the first one, because there only is one
	pdbset = pdblist[0]
	dataset_folder_exists = False
	try:
		os.mkdir(os.path.join(curr_path, 'pdb-dataset'))
	except OSError as e:
		dataset_folder_exists = True
		print("dataset folder already exists")

	pdb_dir = os.path.join(curr_path,'pdb-dataset')
	print(pdb_dir)
	pdb_names = None
	if(dataset_folder_exists != True):
		# pdb_names = extractpdb(os.path.join(dataset_root,pdbset))
		pdb_names = extractpdb(None)
		print(pdb_names[:10])
		fetchpdb(pdb_names,pdb_dir)

	#maybe download failed for a couple of pdb's
	#also getting the full path here for ease of use with the project files
	available_pdbs = [os.path.join(curr_path,'pdb-dataset',item) for item in  os.listdir(pdb_dir)]
	
	#subtract already existing images from the available pdbs
	if(os.path.exists(os.path.join(curr_path,'Images'))):
		#grab all the images which have already been generated
		images = os.listdir(os.path.join(curr_path,'Images'))
		#remove png file ending
		images = [item[:-4] for item in images]
		available_pdbs= [ x for x in available_pdbs if x not in [os.path.join(curr_path,'pdb-dataset',item) for item in images]]


	print(available_pdbs[:10])
	#generating images directory and keeping in absolute
	try:
		os.mkdir(os.path.join(curr_path, 'Images'))
	except OSError as e:
		print('images folder already exists')
	imageoutput  = os.path.join(curr_path,'Images')

	projectoutput = os.path.join(curr_path,'Projects')
	project_dir_existed = False
	try:
		os.mkdir(projectoutput)
	except OSError as e:
		project_dir_existed = True
		print('project folder already exists')
	
	if(project_dir_existed==False):
		# third generate the maps
		for item in available_pdbs:
			generate_project(item,imageoutput,projectoutput,curr_path)

	#megamol needs msms to be in the same path as the binary, it WILL be confused
	#if msms is not there and make some noise
	oldwd = os.getcwd()
	mmdir = os.path.abspath(os.path.join(curr_path,"Maps","bin","x64","Release"))
	os.chdir(os.path.abspath(os.path.join(curr_path,"Maps","bin","x64","Release")))
	#in case someone or something puts something inappropriate in project folder
	for item in list(filter(lambda x: x.endswith('.mmprj'),os.listdir(os.path.abspath(projectoutput)))):
		#yikes
		#this one works
		item = projectoutput+"\\"+item
		# subprocess.call(".\\MegaMolCon.exe " + '-p ' + "C:\\tmp\\comvi\\programming\\image-generation\\Projects\\5uii.mmprj" + ' -i' + ' view1 inst',shell=True)
		pro = subprocess.Popen(".\\MegaMolCon.exe " + '-p ' + item+ ' -i' + ' view1 inst',stdout=open(os.devnull, 'wb'),shell=False)
		try:
			pro.communicate(timeout=30)
		except subprocess.TimeoutExpired as e:
			pro.kill()
			print(str(item))
			print(e)
	#change dir back as if nothing happened 
	os.chdir(oldwd)

if __name__ == '__main__':
	main()
