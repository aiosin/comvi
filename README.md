# comvi
Comparative Visualization of Molecular Surfaces using Similarity-based Clustering

# Megamol-specific terms/keywords:

- *.mmprj: MegaMol Project files
    These xml-based files can be invoked (with specific parameters given by the Configurator) with the megamol-console.

- Configurator: C#/Mono Project files creator, with visual drag and drop interface for project generation.
    MM-Creator creates a xml-file (the *.mmprj files from above) which the megamol can understand.

- StateFile:
	generated by the configurator upon first startup, holds all kinds of information about the mm-binary.
	is nessecary if you want to configure your project with the interface

- 


# Dependencies:
- python:
    - everything was written in python 3.6
    - not tested in  python<3.6 (or 2.7 for that matter)
    - dependencies:
        - numpy
        - tensorflow
        - sklearn
        - skimage
        - scipy
        - cv2
        - matplotlib for 2d plot-viz
- comvi C++:
    - not sure
    - probably only vanilla c++11/14 (?) apart from:
        - glfw
        - glew
        - glad
        - glm
        - std_image
        - stdlib 
        - KHR 
        - opengl

# Folder Structure:
- papers:
    - contains all relevant papers for:
        - clustering
        - viz
        - feature extraction etc. 
- comvi Ausarbeitung:
    - contains `.tex` sourcecode for the comvi paper

- programming:
    - contains:
        - PCA (python):
            - using dimensionality of extracted feature vector to cluster images
            - should be ported to C++ for the final project
        - SSIM (python):
            - uses similarity measure wihout relying on feature vector to cluster images
            - should be ported to C++ for the final project
        - image-generation (python):
            - will be used to generate images for the clustering 
            - performs: pdb aggregation, pdb generation, image generation
        - comvi: 
            - C++ project that will be used as a 2D rendered as a plugin in Megamol

- Images:
    - Contains Images to sample 
    - Delete when image-generation is up and functional for arbitraty pdb datasets
    
# Already Implemented:

- feature extraction

- similarity measure (SSIM, cosine (with feature vector))

- clustering with DBSCAN

- dim. reduction 

- image generation

- image exploration in clusterview 
