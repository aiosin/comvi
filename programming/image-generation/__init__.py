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