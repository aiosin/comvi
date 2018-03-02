
#filesystem related
import os
import sys

#pdb fetching/decoding of json-serialized data
import requests as re
import shutil as sh
import json

#needed for 
import subprocess

#if parallel execution will be a priority at some point
import threading
import multiprocessing
def extractpdb(path=None):
	#current test dataset will have ~1500 imags
	#to reduce computation 
	fileendings = ('.json')
	file = None
	#return value that 
	pdb = []
	#if path not specified take the one from the git repo which unless
	#it has been deleted (by accident I hope) will be there
	if path is None:
		gitroot = os.chdir("../..")
		files = os.listdir(gitroot)
		files = filter(lambda x: x.endswith(fileendings),files)
		files = sorted(files)
		#get the pdbearch file
		files = filter(lambda x: x.contains('pdbsearch'))
		#stub for case that we have multiple files
		if(len(files)> 1):
			file = files[0]
		else:
			file = files[0]
	 else:
		#we have taken the absolute path way, implementation will follow shortly
		#this subroutine should set the file variable to the path
		pass
	if file is not None:
		with open(file,mode='r') as f:
			raw = f.read()
			#file is relatively big, this could be optimized
			json_parsed = json.loads(raw)
			pdb = [key['groupValue'] for key in json_parsed]
	#else branch for debugging purposes
	#with explicit None return
	else:
		print("no file detected")
		return None

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
def fetchpdb(pdblist):
	failed = []
	baseurl = 'https://files.rcsb.org/download/'
	#url for testing purposes in-browser:
	#https://files.rcsb.org/download/4hhb.pdb
	try:
		sh.rmtree('pdb_dataset')
	except FileNotFoundError as e:
		os.mkdir('pdb_dataset')
	os.getcwd()
	#just makin sure we get a list
	if type(pdblist) == type([]):
		for item in pdblist:
			with open(str(item)+'.pdb', mode='wb') as f:
				#try except needed at this point to avoid 404's,
				#and various filesytem related issues
				try:
					resp = re.get(baseurl+str(item)+'.pdb')
					#writing bytestream exactly how we got it
					#no idea if pdb's are ascii or not
					f.write(resp.content)   
				#if exceptions occur we do nothing and quietly continue
				#with the next pdb file, for debugging purposes we
				#keep track of the failures 
				except Exception as e:
					failed.append(item)
				


#TODO: if clean code becomes a priority at some point one should clean this up				
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
	files = filter(lambda x: x.endswith('.pdb'), files)
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
				#bring out the big guns

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


def main():
	#idea of execution for now:
		# First prep the pdb's to fetch
		# second get the pdb's
		# third generate the maps
	pass 

if __name__ == '__main__':
	main()