from skimage import io
from skimage.util import random_noise
from skimage.filters.rank import median, mean
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage.color import label2rgb, rgb2gray
from skimage import measure, img_as_ubyte
from skimage.draw import (line)
from skimage.morphology import disk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import cv2


##add dilation 

def main():
    
     #make a folder called CA for example save this script init
    #and save the photo you want to import in it also then you can use
    #the command below
    scale = 1 #microns/pixel
    sigma_values = [0, 0.2, 0.055, 0.01, 0.02, 0.2]
    output_images = []
    img = readInImage("scissors_col_2.jpg")
    
    imag = convertImageToGreyscale(img)
    
    noised_images = addGaussianNoise(imag, sigma_values)
    print(noised_images[-1])
    
    threshold, thresholded_images = applyOtsuThresholding(noised_images)
   
    plotHistogram(threshold, noised_images, sigma_values)
    
    images_with_borders_removed = clearObjectsTouchingImageBorders(thresholded_images, sigma_values)
    
    region_labelled_images, region_labelled_images_overlays = labelAndColourCodeImageRegions(img, images_with_borders_removed, sigma_values)
    
    ii = 0
    for i in range(len(sigma_values)):
        all_props, props = calculateAndReturnRegionProperties(region_labelled_images[ii], img)
        
        #function that uses pandas to take tha same information and put it in a csv table
        displayImagePropsInPandasTable(props, scale)
        
        #function that takes in the scissors' area and centroids and determines the smallest scissors from
        #using props.suler_number which isolates the image by their unique identifier which is that they
        #are the only objects with two holes. then we use the min(area) from this returns the min scissors and min scissors centroid too
        scissor_area, scissor_centroid_index, centroid_of_smaller_scissors = determineSmallestScissorsAndGetProperties(all_props, thresholded_images[ii])
        
        #lastly using the labels again we use ,atplotlib.mpatches_rectangle to draw a rectangle around the
        #label that has the same area as the smallest scissors. Since we know the scissors area we can 
        #set an if statement to only draw a box around the smallest scissor
        image_with_smallest_scissor = addBoxAroundRegion(thresholded_images[ii], region_labelled_images[ii], region_labelled_images_overlays[ii], scissor_area, scissor_centroid_index, centroid_of_smaller_scissors)
        output_images.append(image_with_smallest_scissor)
        ii += 1
        
    fig, ax = plt.subplots(ncols = len(sigma_values), figsize = (20, 15))
    ii = 0
    for i in range(len(sigma_values)):
        
        ax[i].imshow(noised_images[ii], cmap = plt.cm.gray)
        ii += 1


def readInImage(image1):
    
    img1 = img_as_ubyte(io.imread(image1))
   
    #plt.imshow(img1)
    #plt.axis("off")
    #plt.show()
    
    return img1

#define a function that takes in an image and
#converts it to greyscale
def convertImageToGreyscale(image):
    
    grayscale = rgb2gray(image)
    #plt.imshow(grayscale, cmap=plt.cm.gray)
    #plt.axis("off")
    #plt.show()
    return grayscale

def addGaussianNoise(image, sigma):
    
    #fig, ax = plt.subplots(ncols = 1, nrows= 1, figsize = (10, 8))
    ii = 0
    noised_images = []
    for i in range(len(sigma)):
        
        noise = random_noise(image, mode='gaussian', seed=None, clip=True, var = sigma[ii]**2, mean = 0.0)
        noised_images.append(noise)
            
            
        ii += 1
    
    
   # plt.tight_layout()
    #ax.set_title("nosed kmage")
    return noised_images

def addMedianFilter(image, rank):
    
    #in order fo the mean rank filter to work on our image
    #we need avoid possible precision loss by converting our
    #image which is of type float64 to uint8 this is required by
    #rank filters# we use the command below to achieve this
    noiseconverted = img_as_ubyte(image, force_copy=False)
    medianFilter = median(noiseconverted, disk(rank))
    return medianFilter

def addMeanFilter(image, rank):
    
     #in order fo the mean rank filter to work on our image
    #we need avoid possible precision loss by converting our
    #image which is of type float64 to uint8 this is required by
    #rank filters# we use the command below to achieve this
    noiseconverted = img_as_ubyte(image, force_copy=False)
    meanFilter = mean(noiseconverted, disk(rank))
    
    return meanFilter

def plotHistogram(threshold, image, sigma_values):
    
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (15, 8)) 
    ii = 0
    
    for i in range(0, 2):
        for j in range(0, 3):
        
            
            x = [threshold[ii], threshold[ii]]
            y = [0, 1e6]
            ax[i, j].hist(image[ii].ravel(), bins=50, alpha = 1, color = "blue", label = "Otsu Threshold = {}".format(round(threshold[ii], 3)))
            ax[i, j].plot(0, -1, "ro", markersize = 4, label = "'$\sigma = {}$'".format(sigma_values[ii]))
            ax[i, j].plot(x, y, linewidth = 0.8, linestyle = "dashed", color = "red")
            ax[1, 0].set_xlabel('Intensity Value', fontsize = 13)
            ax[i, j].set_ylabel('Count', fontsize = 13)
            ax[i, j].set_title("Histogram for noised image '$\sigma = {}$'".format(sigma_values[ii]))
            ax[i, j].set_xlim(0, 1)
            ax[i, j].legend(loc="upper right")
            ii += 1
            
   
    fig.suptitle('Threshold Values from Otsu Algorithm on Noised Images', fontsize=20)
    plt.tight_layout()  
    
     #plt.savefig("CA1_out/Q3/Q3_part2b.jpg")
    plt.show()
    
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (15, 8)) 
    ii = 0
    
    for i in range(0, 2):
        for j in range(0, 3):
            
            x = [threshold[ii], threshold[ii]]
            y = [0, 80000]
            ax[i, j].hist(image[ii].ravel(), bins=50, alpha = 1, color = "blue", label = "Otsu Threshold = {}".format(round(threshold[ii], 3)))
            ax[i, j].plot(0, -1, "ro", markersize = 4, label = "'$\sigma = {}$'".format(sigma_values[ii]))
            ax[i, j].plot(x, y, linewidth = 1.4, linestyle = "dashed", color = "red")
            ax[1, 0].set_xlabel('Intensity Value', fontsize = 13)
            ax[i, j].set_ylabel('Count', fontsize = 13)
            ax[i, j].set_title("Histogram for noised image '$\sigma = {}$'".format(sigma_values[ii]))
            ax[i, j].set_xlim(0, 1)
            if(i == 0):
                ax[i, j].set_ylim(0, 2000)
            else:
                ax[i, j].set_ylim(0, 80000)  
                ax[i, j].legend(loc="upper right")
            ii += 1
            
   
    fig.suptitle('Zoomed in histograms to see the accuracy of Otsu', fontsize=20)
    plt.tight_layout()  
    
     #plt.savefig("CA1_out/Q3/Q3_part2b.jpg")
    plt.show()
    
def applyOtsuThresholding(noised_images):
    
    #Generate thresholded image
    image_thresholds = []
    thresholded_images = []
    for i in range(len(noised_images)):
        threshold = threshold_otsu(noised_images[i])
        thresholded_img = (noised_images[i] > threshold)
        
        image_thresholds.append(threshold)
        thresholded_images.append(thresholded_img)
   
    
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (10, 8))
    ii = 0
    for i in range(0, 3):
        for j in range(0, 2):
        
           
            #ax[j, i].imshow(image)
            ax[j, i].imshow(thresholded_images[ii], cmap=plt.cm.gray)
    
            ax[j, i].set_title("({}) Thresholded Imgaes\nthresh = {}$'".format(ii + 1, round(image_thresholds[ii], 3)), fontsize = 14)
            #ax[i, j].set_title("image with gaussian noise (sigma = {})".format(sigma[ii]), fontsize = 14)
    
            ax[j, i].axis("off")
            ax[j, i].axis("off")
            
            #if(i == 1 and j == 2):
             #   print(noise)
            
            ii += 1
    
    
    plt.tight_layout()
    
    return image_thresholds, thresholded_images
 
    
def clearObjectsTouchingImageBorders(thresholded_images, sigma_values):
    
    #Remove edge touching regions
    fig, ax = plt.subplots(ncols = 3, nrows= 2, figsize = (10, 8)) 
    ii = 0
    images_with_borders_removed = []
    for i in range(len(thresholded_images)):
        
        edge_touching_removed = clear_border(thresholded_images[ii])
        images_with_borders_removed.append(edge_touching_removed)
        ii += 1 
    
    ii = 0
    for i in range(0, 2):
         for j in range(0, 3):
        
            ax[i, j].imshow(images_with_borders_removed[ii], cmap = plt.cm.gray)
    
            ax[i, j].set_title("Thresholded Image {} with\nBorders Removed '$\sigma = {}$'".format(ii + 1, sigma_values[ii]), fontsize = 12)
            #ax[i, j].set_title("image with gaussian noise (sigma = {})".format(sigma[ii]), fontsize = 14)
    
            ax[i, j].axis("off")
            ax[i, j].axis("off")
            
            ii += 1
    
    fig.suptitle('Border Objects Removed from Thresholded images of varying noise', fontsize=20)
    plt.tight_layout()
    plt.imshow(edge_touching_removed, cmap=plt.cm.gray)
    
    return images_with_borders_removed

def labelAndColourCodeImageRegions(image, images_with_borders_removed, sigma_values):
    
    #Label connected regions of an integer array using measure.label
    #Labels each connected entity as one object
    #Connectivity = Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
    #If None, a full connectivity of input.ndim is used, number of dimensions of the image
    #For 2D image it would be 2
    fig, ax = plt.subplots(ncols = 3, nrows= 2, figsize = (10, 8)) 
    ii = 0
    region_labelled_images = []
    region_labelled_images_overlays = []
    for i in range(len(images_with_borders_removed)):
        
        region_labelled_image = measure.label(images_with_borders_removed[ii])
        region_labelled_images.append(region_labelled_image)
        #plt.imshow(label_image)
    
    
        #Return an RGB image where color-coded labels are painted over the image.
        #Using label2rgb
        region_labelled_image_overlay = label2rgb(region_labelled_image, image=image)
        region_labelled_images_overlays.append(region_labelled_image_overlay)
        #plt.imshow(image_label_overlay)
        #plt.imsave("labeled_cast_iron.jpg", image_label_overlay)
        ii += 1
    
    ii = 0
    for i in range(0, 2):
        for j in range(0, 3):
        
            ax[i, j].imshow(region_labelled_images_overlays[ii])
    
            ax[i, j].set_title("Region labelled image {}'$\sigma = {}$'".format(ii + 1, sigma_values[ii]), fontsize = 12)
            #ax[i, j].set_title("image with gaussian noise (sigma = {})".format(sigma[ii]), fontsize = 14)
    
            ax[i, j].axis("off")
            ax[i, j].axis("off")
            
            ii += 1
    
    fig.suptitle('Border Objects Removed from Thresholded images of varying noise', fontsize=20)
    plt.tight_layout()
    
    return region_labelled_images, region_labelled_images_overlays



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

    dataFrame = pd.DataFrame(props)
    print(dataFrame.head())

    #To delete small regions...
    dataFrame = dataFrame[dataFrame['area'] > 50]
    print(dataFrame.head())

    #######################################################
    #Convert to micron scale
    dataFrame['area_sq_microns'] = dataFrame['area'] * (scale**2)
    dataFrame['equivalent_diameter_microns'] = dataFrame['equivalent_diameter'] * (scale)
    print(dataFrame.head())

    #dataFrame.to_csv('CA1_out/Q2/cast_iron_measurements.csv')    
 
def determineSmallestScissorsAndGetProperties(all_props, thresholded_img):
    
     #set variables to hold the num of scissors, the area and centroids
     num_scissors = 0
     scissor_centroids = []
     scissor_areas = []
     
     #we can use euler_number from our image regions to detect objects
     #with two holes aka the scissor objects
     for props in all_props:  # one RegionProps object per region

         #to detect objects with two holes set euler number == -1
            if props.euler_number == -1:
                
                #if scissor has been detected append and append area and centroid
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
     
      
     print("\nthe num of scissors is: ", num_scissors)
     print("\nthe smaller scissors has an area of: ", area_of_smaller_scissors)
     print("\nthe centroid of the smaller scissors is: ", centroid_of_smaller_scissors)
     print("centroid index is ", centroid_index)
     
     return area_of_smaller_scissors, centroid_index, centroid_of_smaller_scissors


def addBoxAroundRegion(thresholded_img, label_image, image_label_overlay, scissor_area, scissor_centroid_index, centroid_of_scissor):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    print(label_image.shape)
    
    print("this issssss", centroid_of_scissor)

    #as we have detemined the min scissors' area in the last function we can
    #use this as a condition on whetehr or not to draw a box around our region labels
    #so for all of our regions we only draw a box if the regions area is equal to the area
    #of the min scissors
    for region in regionprops(label_image):
    # take regions with large enough areas
        if region.area == scissor_area:
        # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(rect)
            
    
    #convert our labelled image to greyscale for the output plot
    image_label_converted_to_grey_scale = rgb2gray(label_image)
   
    #image_label_converted_to_grey_scale = threshold_otsu(image_label_converted_to_grey_scale)
    #image_label_converted_to_grey_scale = clear_border(label_image)
    ax.imshow(image_label_converted_to_grey_scale, cmap=plt.cm.gray)
    
     #using skimage.draw we can use the centroid infromation to draw the cordinates of the
     #smallest scissor centroid on the image directly
    x_coord1, y_coord1 = line(0, int(centroid_of_scissor[1]), image_label_converted_to_grey_scale.shape[0] - 1, int(centroid_of_scissor[1]))
    x_coord2, y_coord2 = line(int(centroid_of_scissor[0]), 0, int(centroid_of_scissor[0]), image_label_converted_to_grey_scale.shape[1] - 1)
    image_label_converted_to_grey_scale[x_coord1, y_coord1] = 15
    image_label_converted_to_grey_scale[x_coord2, y_coord2] =15
    ax.imshow(image_label_converted_to_grey_scale, cmap=plt.cm.gray)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    #print(thresh)
    
    return image_label_converted_to_grey_scale
    
 
main()
