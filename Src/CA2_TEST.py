# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:13:36 2021

@author: evanMcGraneTest
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage import measure, io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray
import numpy as np
import pandas as pd
from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)


import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

def main():
    
     #make a folder called CA for example save this script init
    #and save the photo you want to import in it also then you can use
    #the command below
    scale = 0.6 #microns/pixel
    img = readInImage("scissors_col_2.jpg")
    
    image = convertImageToGreyscale(img)
    
    #plotHistogram(image) ##plt.hist(blue_channel.flat, bins=100, range=(0,150))  #.flat returns the flattened numpy array (1D)
    
    threshold, thresholded_img = applyOtsuThresholding(image)
    
    edge_touching_removed = clearObjectsTouchingImageBorders(thresholded_img)
    
    label_image, image_label_overlay = labelAndColourCodeImageRegions(image, edge_touching_removed)
    
    all_props, props = calculateAndReturnRegionProperties(label_image, image)
    
   
    displayImagePropsInPandasTable(props, scale)
    
     
    scissor_area, scissor_centroid_index, centroid_of_smaller_scissors = determineSmallestScissorsAndGetProperties(all_props, thresholded_img)
    
    addBoxAroundRegion(label_image, image_label_overlay, scissor_area, scissor_centroid_index, centroid_of_smaller_scissors)

    print(thresholded_img.shape[1] - 1)




def readInImage(image1):
    
    img1 = img_as_ubyte(io.imread(image1))
   
    
    plt.imshow(img1)
    plt.axis("off")
    plt.show()
    
    return img1

#define a function that takes in an image and
#converts it to greyscale
def convertImageToGreyscale(image):
    
    grayscale = rgb2gray(image)
    plt.imshow(grayscale, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()
    return grayscale

def plotHistogram(image):
    
    fig, ax = plt.subplots(facecolor ='#A0F0CC')
    ax.hist(image.ravel(), bins =256, alpha = 1)  #.flat returns the flattened numpy array (1D)
    ax.set_ylim(0, 8000)
    plt.show()
    
def applyOtsuThresholding(image):
    
    #Generate thresholded image
    threshold = threshold_otsu(image)
    thresholded_img = (image > threshold)
    plt.imshow(thresholded_img, cmap=plt.cm.gray)
    plt.axis("off")
    
    return threshold, thresholded_img
 
    
def clearObjectsTouchingImageBorders(thresholded_img):
    
    #Remove edge touching regions
    edge_touching_removed = clear_border(thresholded_img)
    plt.imshow(edge_touching_removed, cmap=plt.cm.gray)
    
    return edge_touching_removed

def labelAndColourCodeImageRegions(image, edge_touching_removed):
    
    #Label connected regions of an integer array using measure.label
    #Labels each connected entity as one object
    #Connectivity = Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
    #If None, a full connectivity of input.ndim is used, number of dimensions of the image
    #For 2D image it would be 2
    
    label_image = measure.label(edge_touching_removed, connectivity=image.ndim)
    plt.imshow(label_image)
    
    
    #Return an RGB image where color-coded labels are painted over the image.
    #Using label2rgb
    image_label_overlay = label2rgb(label_image, image=image)
    plt.imshow(image_label_overlay)
    plt.imsave("labeled_cast_iron.jpg", image_label_overlay)
    
    return label_image, image_label_overlay





def calculateAndReturnRegionProperties(label_image, image):

    #Calculate properties Using regionprops or regionprops_table
    all_props = measure.regionprops(label_image, image)
    #Can print various parameters for all objects
    for prop in all_props:
        print('Label: {} Area: {}'.format(prop.label, prop.area))

    #Compute image properties and return them as a pandas-compatible table.
    #Available regionprops: area, bbox, centroid, convex_area, coords, eccentricity,
    # equivalent diameter, euler number, label, intensity image, major axis length, 
    #max intensity, mean intensity, moments, orientation, perimeter, solidity, and many more
    props = measure.regionprops_table(label_image, image, properties=['label','area', 'equivalent_diameter','mean_intensity', 'solidity'])
   
    
    return all_props, props

def displayImagePropsInPandasTable(props, scale):

    df = pd.DataFrame(props)
    print(df.head())

    #To delete small regions...
    df = df[df['area'] > 50]
    print(df.head())

    #######################################################
    #Convert to micron scale
    df['area_sq_microns'] = df['area'] * (scale**2)
    df['equivalent_diameter_microns'] = df['equivalent_diameter'] * (scale)
    print(df.head())

    #df.to_csv('CA1_out/Q2/cast_iron_measurements.csv')    
 
def determineSmallestScissorsAndGetProperties(all_props, thresholded_img):
    
     num_scissors = 0
     scissor_centroids = []
     scissor_areas = []
     
     for props in all_props:  # one RegionProps object per region
            if props.euler_number == -1:
            
                num_scissors += 1
            
                scissor_centroids.append(props.centroid)
                scissor_areas.append(props.area)
                
            
      #now i get the min value in the list to get smallers area and also 
     #the index of the min value in order to use it to get the equvelent
     #centroid in the other list by index
     area_of_smaller_scissors = min(scissor_areas)
        
     #get index of smallerst area as the smallest centroid will be in
     #the same index in its list
     centroid_index = scissor_areas.index(area_of_smaller_scissors)
     centroid_of_smaller_scissors = scissor_centroids[centroid_index]
     
     
     rr, cc = line(0, int(scissor_centroids[centroid_index][1]), thresholded_img.shape[0] - 1, int(scissor_centroids[centroid_index][1]))
     r1r1, c1c1 = line(int(scissor_centroids[centroid_index][0]), 0, int(scissor_centroids[centroid_index][0]), thresholded_img.shape[1] - 1)
     thresholded_img[rr, cc] = 50
     thresholded_img[r1r1, c1c1] =50
     #plt.imshow(thresholded_img)
      
     print("\nthe num of scissors is: ", num_scissors)
     print("\nthe smaller scissors has an area of: ", area_of_smaller_scissors)
     print("\nthe centroid of the smaller scissors is: ", centroid_of_smaller_scissors)
     print("centroid index is ", centroid_index)
     
     return area_of_smaller_scissors, centroid_index, centroid_of_smaller_scissors


def addBoxAroundRegion(label_image, image_label_overlay, scissor_area, scissor_centroid_index, centroid_of_scissor):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    print(label_image.shape)
    
    print("this issssss", centroid_of_scissor)

    for region in regionprops(label_image):
    # take regions with large enough areas
        if region.area == scissor_area:
        # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
    image_label = label_image
    #image_label = rgb2gray(image_label)
    print(image_label.shape)
    #image_label = threshold_otsu(image_label)
    #image_label1 = clear_border(label_image)
    #ax.imshow(image_label)
    
    rr, cc = line(0, int(centroid_of_scissor[1]), image_label.shape[0] - 1, int(centroid_of_scissor[1]))
    r1r1, c1c1 = line(int(centroid_of_scissor[0]), 0, int(centroid_of_scissor[0]), image_label.shape[1] - 1)
    
    image_label[rr, cc] = 10
    image_label[r1r1, c1c1] =10
    ax.imshow(image_label, cmap=plt.cm.gray)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    #print(thresh)
    
    
main()