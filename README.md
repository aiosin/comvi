# comvi
Comparative Visualization of Molecular Surfaces using Similarity-based Clustering


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