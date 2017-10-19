# Face Morphing

## Prerequisites
This project was tested with
- python 3.6.1, version >= 3.0 should work
- matplotlib, used for plotting images/meshes/etc
- scipy, used for the Delaunay triangulation
- numpy, used for most calculations
- pillow, used to load the image files

The included code is broken up into the following files
- face_morph.py, the entrance point, handles argument parsing/showing results
- image.py, code to load an image file, selecting points on the mesh
- mesh.py, Handles saving/loading meshes, performs the actual mesh/image interpolation
- utils.py, misc other code

## Usage
The most basic usage is
- python face_morph.py --image1 SRCFILE --image2 DSTFILE

Loads the file given by SRCFILE/DSTFILE and looks for corresponding mesh files by the name of srcname.mesh where SRCFILE = srcname.(jpg|png|...).
If no mesh file is found it prompts the user to draw points on the image. Once this is done it will output the (default) 5 internal points and end points for the interpolation.

Meshes for each phase of interpolation can be shown if --show-mesh is specified. This will also plot the mesh points on the image

In order to update an existing mesh, the --update-mesh param can be used

The number of internal points can be changed by specifying --internal-points=N

Specifying --save-files will cause all plots of images/meshes to be saved

## Included Samples

A number of test files are included

- obama.jpg, the face of President Obama
- obama.mesh, a mesh of the face of Obama, uses 40 points on the face

- trump.jpg, the face of President Trump
- trump.mesh, a mesh of the face of Trump, order matches with the mesh of Obama

- orange.jpg, a picture of a slice of an orange
- orange.mesh, a mesh of the orange with mesh points done in concentric circles

- orange_face.jpg, the same picture as orange.jpg
- orange_face.mesh, a mesh of the orange with points matching with the Obama/Trump mesh

- obama_warp0.png/trump_warp1.png, the warped faces corresponding to obama/trump.jpg with alpha=0/1 respectively
