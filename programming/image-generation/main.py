import os
import sys
import threading
import multiprocessing

import json

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
    pass
#use https://www.rcsb.org/pages/download/http
#to download pdb's to prepare the "pipeline"
#for the next step
def fetchpdb(pdblist):
    if type(pdblist) == type([]):
        pass

def main():
    pass

if __name__ == '__main__':
    main()