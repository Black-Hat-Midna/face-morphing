# -*- coding: utf-8 -*-
"""

This program will take in images from the images folder within the same current folder
and in that folder should be 2 images face1.jpg and face2.jpg. Face1 will then be 
morphed into face2 and the resulting video will be saved in the program folder
and called morph.mp4.
There are also some test functions that have been commented out in the code
if you wish to view the intermediate results of the algorithm (these functions 
                                    and code blocks will be marked accordingly).
"""

# complete imports and open/create all files needed for image morphing
from scipy.spatial import Delaunay # for working out mesh
import numpy as np # for mathmatical operations
import matplotlib.pyplot as plt # for ploting images
from skimage import io # for reading images
from skimage import transform as tf # for affine warp
from skimage import draw as d # for getting polygons
import cv2 as cv # for saving the final video

wStep = 0.02 # 50 in between images (inc. last)

################
# code block below taken and amended from documentation of https://pypi.org/project/face-alignment/
# for getting points on each face (I.E landmark detection) 

import face_alignment
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#face 1 from https://pxhere.com/en/photo/998102
#face 2 from https://pxhere.com/en/photo/367907
# image1 is public domain
# image2 is creative commons and by Tom Marvel the license for which is below
# https://creativecommons.org/licenses/by/2.0/legalcode
input1 = io.imread('./images/face1.jpg')/255.
input2 = io.imread('./images/face2.jpg')
input2 = tf.resize(input2, (input1.shape[0], input1.shape[1]))

# print(input1.shape)
# print(input2.shape)

# get landmarks using neural network model
# WARNING it assumes you have a good GPU. If you want to run it without a GPU
# you need to open the model running on CPU instead. change device to cpu
# in FaceAlignment func.
print("processing images...")
preds1 = model.get_landmarks(input1*255.)
preds2 = model.get_landmarks(input2*255.)
#print(input2.shape)
#print(input1.shape)
################

############### code block below altered from 
#https://numpy.org/doc/stable/reference/generated/numpy.save.html
with open("preds1.npy", "wb") as file:
    np.save(file, preds1)
with open("preds2.npy", "wb") as file:
    np.save(file, preds2)

with open("preds1.npy", "rb") as file:
    preds1 = np.load(file)
with open("preds2.npy", "rb") as file:
    preds2 = np.load(file)
################

def plotImg(image, points=None):
    """

    Parameters
    ----------
    image : numpy array
        The image the points should be plotted over
    points : numpy array
        Of the shape (num of points, dimensions) dimensions is assumed to be 2
        and represents the points over the image in question.

    Returns
    -------
    None.
    But causes a graph to be plotted containing the points over the image if
    points are set, else just the image.

    """
    
    plt.imshow(image)
    if(points!=None):
        plt.scatter(points[:, 0], points[:, 1], 1)
    plt.show()
    
    return None


def addPointsToPredictions(image, preds):
    """
    
    Parameters
    ----------
    image : numpy array
        The image which is being operated on
    preds : numpy array
        The predictions for a single face from a face landmark detection model

    Returns
    -------
    newPreds : numpy array
        The new prediction array but with additional points to cover the 
        entire image

    Description
    -------
    Adds additional points to the facial landmark detection model in order to 
    make the predictions cover the entire image.
    
    """
    
    newPreds = preds.copy()
    shape = image.shape
    newPreds = np.append(newPreds,  [[0, 0]], axis=0)
    newPreds = np.append(newPreds, [[shape[1]-1, shape[0]-1]], axis=0)

    x_num = 4
    for i in range(1, x_num+1):
        newPreds = np.append(newPreds,  [[int(i*(shape[1]/x_num))-1, 0]], axis=0)
        newPreds = np.append(newPreds,  [[int(i*(shape[1]/x_num))-1, shape[0]-1]], axis=0)
        
    y_num = 4
    for i in range(1, x_num+1):
        newPreds = np.append(newPreds, [[0, int(i*(shape[0]/y_num))-1]], axis=0)
        newPreds = np.append(newPreds, [[shape[1]-1, int(i*(shape[0]/y_num))-1]], axis=0)
        
    return newPreds

def estimateWarpParameters(src, dst):
    """

    Parameters
    ----------
    src : A numpy array of the shape (3, 2)
        3 points on the original or source image.
        Points should be in the form [x, y]
    dst : A numpy array of the shape (3, 2)
        3 points to warp to I.E the new points on the destination image.
        Points should be in the form [x, y]

    Returns
    -------
    T2 : A numpy array of the shape (3, 3)
        Parameters for Affine warp.
        
    Description
    -----------
    made with reference to chapter 15 "Computer vision: models, learning and 
    inference Simon J.D. Prince".
    
    Honestly there are better ways to do this (standardwise and implementation) but this works 
    for my purposes.
    
    """
    
    A = np.array([[src[0, 0], src[0, 1], 1, 0, 0, 0], 
     [0, 0, 0, src[0, 0], src[0, 1], 1],
     [src[1, 0], src[1, 1], 1, 0, 0, 0], 
     [0, 0, 0, src[1, 0], src[1, 1], 1],
     [src[2, 0], src[2, 1], 1, 0, 0, 0], 
     [0, 0, 0, src[2, 0], src[2, 1], 1]])
    B = np.array([[dst[0, 0]], [dst[0, 1]], [dst[1, 0]], 
                  [dst[1, 1]], [dst[2, 0]], [dst[2, 1]]])
    T = np.linalg.inv(A.T@A)@A.T@B # normal equation for linear regression
    #again better ways to do this but this works well.
    
    T2 = np.zeros((3, 3))
    T2[0:2, :] = T.reshape((2, 3))
    T2[2, 2] = 1
    return T2

def affineTransform(image1, image2, tri, transforms1, transforms2, lin):
    """

    Returns
    -------
    warpedImage : A numpy array of floats
        An image which has been warped according to the given triangle mesh,
        interpolated points (lin) and the transforms for image1 (transforms1). 
        The intensities are also based on image1 and bilinearly 
        interpolated.
    warpedImage : A numpy array of floats
        An image which has been warped according to the given triangle mesh,
        interpolated points (lin) and the transforms for image2 (transforms2). 
        The intensities are also based on image2 and bilinearly 
        interpolated.
        
    """

    warpedImage1 = np.zeros(image2.shape)
    warpedImage = np.zeros(image1.shape)
    for i in range(0, len(tri.simplices)):
        #we need to seperate the components of each image (i.e each triangle) 
        #and warp to our inbetween point based on w (it will be im1 -> im2)
        
        #warp image to new point for one polygon
        warpPoly = warpImage(image1, transforms1[i])##tf.warp(image1, np.linalg.inv(transforms1[i]))lib versions!!
        warpPoly1 = warpImage(image2, transforms2[i])##tf.warp(image2, np.linalg.inv(transforms2[i]))
        
        #polygon2mask takes in coords as [y,x] although not specified in docs
        poly = np.append(lin[tri.simplices[i]][:, 1:2], lin[tri.simplices[i]][:, 0:1], axis=-1)
        mask1 = d.polygon2mask(image1.shape, poly)
        polyImage1 = warpPoly*mask1 #get only polygon of warp from image
        
        mask2 = d.polygon2mask(image2.shape, poly)
        polyImage2 = warpPoly1*mask2 #get only polygon of warp from image
        
        #then we need to find the intensity at each pixel given the transformations
        #depending on your image sizes and affine transforms, transform cannot be perfect
        #given an image is discrete so just pick a poly to match to zone
        
        ## test block shows polys warped, parts of which are based on code found in the scipy documentation 
        # refer to reference at the bottom of the codebase
        # plt.imshow(polyImage1)
        # plt.triplot(lin[:,0], lin[:,1], tri.simplices)
        # plt.plot(lin[:,0], lin[:,1], ',')
        # plt.show()
        
        ind1 = np.where((polyImage1>0) & (warpedImage>0))
        ind2 = np.where((polyImage2>0) & (warpedImage1>0))
        
        polyImage1[ind1] = 0
        polyImage2[ind2] = 0
        
        #add poly to image
        warpedImage += polyImage1
        warpedImage1 += polyImage2
    
    ## test funcs shows warped images
    ## parts of which are again based on lines found in the scipy documentation 
    # plt.imshow(warpedImage)
    # #plt.triplot(lin[...,0], lin[...,1], tri.simplices)
    # #plt.plot(lin[...,0], lin[...,1], ',')
    # plt.show()
    
    # plt.imshow(warpedImage1)
    # #plt.triplot(lin[...,0], lin[...,1], tri.simplices)
    # #plt.plot(lin[...,0], lin[...,1], ',')
    # plt.show()

    return warpedImage, warpedImage1

def affineTransformMapper(tri, im1P, im2P, lin):
    """
    
    Parameters
    ----------
    tri : tri
        contains the polygons.
    im1P : numpy array of points of the shape (points, 2) each like [x,y] 
        landmark points for img1
    im2P : numpy array of points of the shape (points, 2) each like [x,y] 
        landmark points for img2
    lin : numpy array of points of the shape (points, 2) each like [x,y] 
        landmark points of linearly interpolated points between img1 and img2

    Returns
    -------
    params1 : list of numpy arrays which are of shape (3,3)
        Since the affine transform is for each triangle we map each triangle of
        the first image onto the second image. params is simply the worked out
        transformation parameters for each triangle.
    params2 : see above but for im2P

    """

    params1 = []
    params2 = []
    for i in range(0, len(tri.simplices)):
        
        trans = estimateWarpParameters(im1P[tri.simplices[i]], lin[tri.simplices[i]])
        trans1 = estimateWarpParameters(im2P[tri.simplices[i]], lin[tri.simplices[i]])
        
        # when using the library instead you can uncomment block below and can comment other code
        # in the for loop
        
        # trans = tf.AffineTransform()
        # trans.estimate(im1P[tri.simplices[i]], lin[tri.simplices[i]])
        # trans1 = tf.AffineTransform()
        # trans1.estimate(im2P[tri.simplices[i]], lin[tri.simplices[i]])
        # params1.append(trans.params)
        # params2.append(trans1.params)
        
        params1.append(trans)
        params2.append(trans1)
        
    return params1, params2

def interpolateLinearly(im1p, im2p, w):
    #Made based on information obtained from https://en.wikipedia.org/wiki/Linear_interpolation
    deltax = (1-w)*im1p[..., 0] + w*im2p[..., 0]
    deltay = (1-w)*im1p[..., 1] + w*im2p[..., 1]
    
    return np.append(deltax[:, np.newaxis], deltay[:, np.newaxis], axis=-1)

def colorInterpolateLinearly(img1, img2, w):
    #again refer to above for reference
    fImg = (1-w)*img1 + img2*w
    return fImg

def convertToVideo(imgs, name):
    try:
        write = cv.VideoWriter(name+".mp4", cv.VideoWriter_fourcc('m', 'p', '4', 'T'), 10, (imgs[0].shape[1], imgs[0].shape[0]))
        for i in imgs:
            write.write(cv.cvtColor(np.round((i*255).astype(np.uint8)), cv.COLOR_RGB2BGR))
        write.release()
        return True
    except Exception:
        pass
    return False

def warpImage(image, params):
    """
    
    Description
    -----------
    
    inversly warp the given image given the provided set of parameters and then 
    uses bilinear interpolation to sample the source image. Follows affine 
    transform equations found in chapter 15 "Computer vision: models, learning 
    and inference Simon J.D. Prince" again.
    
    """
    
    #get image as coordinates backward remember
    src = np.ones((image.shape[0]*image.shape[1], 3))

    #so I worked out the params in the form of [X,Y] so switch image shape for calc
    #since image is actually [Y,X]
    imagePoints = np.indices((image.shape[1], image.shape[0])).reshape(2, -1).T
    src[:, 0:2] = imagePoints
    
    # now calculate new points transposing params as src is row not column
    b = np.dot(src, np.linalg.inv(params).T)
    b = b/b[:, 2:3] # line technically not needed
    
    imageOut = np.zeros((image.shape[0], image.shape[1], 3))
        
    # find where points match in both the dst (although I've named it src) image and the transformed image
    indT = np.isin(np.round(b[..., 0:1]).astype(np.int64), src[..., 0:1])
    indI = np.isin(np.round(b[..., 1:2]).astype(np.int64), src[..., 1:2])
    #imageOut[imagePoints[ind, 1], imagePoints[ind, 0], :] = image[b[ind, 1].astype(np.int64), b[ind, 0].astype(np.int64), :] # this works as a quick and dirty test
    
    ind = np.where(indT&indI)[0]
    #Y,X
    #find points for bilinear interpolation
    p1 = np.array([np.floor(b[ind, 1]), np.floor(b[ind, 0])]).astype(np.int64)
    p2 = np.array([np.ceil(b[ind, 1]), np.floor(b[ind, 0])]).astype(np.int64)
    p3 = np.array([np.floor(b[ind, 1]), np.ceil(b[ind, 0])]).astype(np.int64)
    p4 = np.array([np.ceil(b[ind, 1]), np.ceil(b[ind, 0])]).astype(np.int64)
    
    #clip the points which could be rounded to the higher or lower than image size
    #delta should handle where x,y == x+1,y due to clipping
    p1[0] = np.clip(p1[0], 0, image.shape[0]-1)
    p1[1] = np.clip(p1[1], 0, image.shape[1]-1)
    p2[0] = np.clip(p2[0], 0, image.shape[0]-1)
    p2[1] = np.clip(p2[1], 0, image.shape[1]-1)
    p3[0] = np.clip(p3[0], 0, image.shape[0]-1)
    p3[1] = np.clip(p3[1], 0, image.shape[1]-1)
    p4[0] = np.clip(p4[0], 0, image.shape[0]-1)
    p4[1] = np.clip(p4[1], 0, image.shape[1]-1)

    #find delta for interpolation
    ind1 = np.array([b[ind, 1], b[ind, 0]])
    delta = ind1 - p1

    #bilinear interpolation of colors
    p1I = image[p1[0], p1[1]] * (1-delta[0, :, np.newaxis]) * (1-delta[1, :, np.newaxis])
    p2I = image[p2[0], p2[1]] * (delta[0, :, np.newaxis]) * (1-delta[1, :, np.newaxis])
    p3I = image[p3[0], p3[1]] * (1-delta[0, :, np.newaxis]) * (delta[1, :, np.newaxis])
    p4I = image[p4[0], p4[1]] * (delta[0, :, np.newaxis]) * (delta[1, :, np.newaxis])
    
    imageOut[imagePoints[ind, 1], imagePoints[ind, 0], :] = np.clip(p1I+p2I+p3I+p4I, 0, 1) # account for rounding errors
    return imageOut

######################
#print(np.array(preds1).shape, ": An example shape outputted by the facial landmark detection neural network")
#the 3 in the shape is the number of predictions we can basically ignore any but the first set

#####################
## plotting
#An example of the used face align model
#note to understand my predictions from the model 
#I looked at https://github.com/1adrianb/face-alignment for their examples
#the github doubles as their documentation so I need to use it!!!
#we can assume there is only one face in the image or at least one face that we care about so we can do preds[0] and 
#from my understanding results will be ordered by confidence ergo index 0
plt.imshow(input1)
plt.scatter(preds1[0, :, 0], preds1[0, :, 1], 1)
plt.show()

plt.imshow(input2)
plt.scatter(preds2[0, :, 0], preds2[0, :, 1], 1)
plt.show()

######################
#Add points to the prediction as the linear intopolation needs to cover the whole image
face1 = preds1[0]
face2 = preds2[0]
face1 = addPointsToPredictions(input1, face1)
face2 = addPointsToPredictions(input2, face2)

######################
#This block of code is based on the scipy documentation and amended for my case
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html 
#I assume this is what you intended us to do (the test blocks are adapted from the docs as well)

tri = Delaunay(face1, False)

# ## plotting/test methods
# plt.imshow(input1)
# plt.triplot(face1[...,0], face1[...,1], tri.simplices)
# plt.plot(face1[...,0], face1[...,1], ',')
# plt.show()

# ## plotting/test methods
# plt.imshow(input2)
# plt.triplot(face2[...,0], face2[...,1], tri.simplices)
# plt.plot(face2[...,0], face2[...,1], ',')
# plt.show()

######################

plt.imshow(input1)
plt.show()

outputImages = []
outputImages.append(input1)
vid = plt.figure()
for i in range(1, 50):
    w = i*0.02
    print(i)
    print("Processing for w: "+str(w))
    lin = interpolateLinearly(face1, face2, w)
    (transParams1, transParams2) = affineTransformMapper(tri, face1, face2, lin)
    im0, im1 = affineTransform(input1, input2, tri, transParams1, transParams2, lin)
    finalImage = colorInterpolateLinearly(im0, im1, w)

    outputImages.append(finalImage)
    
    ###test block
    # again refer to scipy reference above for triplot lines.
    # plt.imshow(input1)
    # plt.triplot(lin[...,0], lin[...,1], tri.simplices)
    # plt.plot(lin[...,0], lin[...,1], ',')
    # plt.show()
    
    #plt.imshow(im1)
    #plt.show()

    #plt.imshow(im0)
    #plt.show()

    plt.imshow(finalImage)
    plt.show()
    
plt.imshow(input2)
plt.show()
    
outputImages.append(input2)

if(convertToVideo(outputImages, "morph")):
    print("video outputted as morph.mp4")
else:
    print("file was unable to be saved")