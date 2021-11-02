from skimage import io
from skimage.util import random_noise
from skimage.filters.rank import median, mean
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import clear_border
from skimage.measure import regionprops,  regionprops_table
from skimage.color import label2rgb, rgb2gray
from skimage import measure, img_as_ubyte
from skimage.draw import (line)
from skimage.morphology import disk, closing, square
import matplotlib.pyplot as plt


def main():
    
    #this global array stores various sigman 
    #values that we are going to pass into the 
    #scissor detection algorithm
    sigma_values = [0.0, 0.01, 0.04, 0.06, 0.08, 0.5]
    
    #here we declare a few global arrays that we will
    #be using for plotting our resutls at the end
    small_sciss_list = []
    noised_images = []
    image_thresholds = []
    dialated_images = []
    imagesBordersRemoved = []
    
    #read in image
    img = readInImage("scissors_col_2.jpg")
    
    #convert to greyscale
    imag = convertImageToGreyscale(img)
    
    #since we needed to test our algorithms 
    #effictivness the scissor detection algorithm 
    #is wrapped in a loop which runs the algorithm
    #for various noise levels
    for i in range(len(sigma_values)):
    
        #function that adds noise and appends the 
        #noised image to an array
        #for plotting
        noise = addGaussianNoise(imag, sigma_values[i], 
                                 sigma_values)
        noised_images.append(noise)
        
        #function that a mean filter to reduce noise
        #and appends the result
        #to an array for plotting
        noise1 = addMedianFilter(noise, 8)
    
        #function that applies an Otsu threshold in order to 
        #seperate the image background and foeground. thresholded 
        #image appened to an array for plotting
        threshold, thresh_image = OtsuThresh(noise1)
        image_thresholds.append(threshold)
        
        #here we dilate the image to make the detection 
        #algorithm preform better for hgher levels of noise
        #append result to an array for plotting
        dialated_img = thresh_image
        dialated_images.append(dialated_img)
   
     #   plotHistogram(threshold, noised_images, sigma_values)
        
        #here we remove objects touching the 
        #border as we do not want to include these
        #objects in the detction algotithm. 
        #appened border removed images to array
        borders_removed = clearBorder(dialated_img, sigma_values)
        imagesBordersRemoved.append(borders_removed)
        
        #here we call a function that labels all of the 
        #objects in the image so that we can do further 
        #processing. append labelled image to array for plotting
        labelled_img, overlay, lb1 = LabelRegions(img, borders_removed, 
                                             sigma_values)
    
      
        #this last algorithm calculates the images 
        #object properties and uses this to isolate the 
        #smaller scissors. result appended to array
        #for pltting
        small_sciss, props  = calculateRegionProps(labelled_img, img, 
                                                   sigma_values, i)
        small_sciss_list.append(small_sciss)
        
   
    #when the scissor detection algorithm finishes for all noised
    #images we plot the results
    plotNoisedImages(imag, noised_images)
    plotClearBorderImages(imagesBordersRemoved)
    plotRegionLabelledImages(dialated_images, sigma_values)
    plotIsolatedScissors(small_sciss_list)
    plotAllSteps(img, imag, noise, thresh_image, borders_removed, lb1, overlay, small_sciss )
    

def plotAllSteps(img, imag, noise, thresh_image, borders_removed, lb1, overlay, small_sciss):
    
     fig, ax = plt.subplots(ncols = 4, nrows=2, figsize = (10, 8))
     ax[0,0].imshow(img)
     ax[0,1].imshow(imag, cmap = plt.cm.gray)
     ax[0,2].imshow(noise, plt.cm.gray)
     ax[0,3].imshow(thresh_image)
     ax[1,0].imshow(borders_removed)
     ax[1,1].imshow(lb1)
     ax[1,2].imshow(overlay)
     ax[1,3].imshow(small_sciss)
     ax[0, 0].set_title("(1) Original image")
     ax[0, 1].set_title("(2) Greyscale image")
     ax[0, 2].set_title("(3) Noise added")
     ax[0, 3].set_title("(4) Apply Otsu threshold")
     ax[1, 0].set_title("(5) clear image borders")
     ax[1, 1].set_title("(6) label image regions")
     ax[1, 2].set_title("(7) image labelled region overlay")
     ax[1, 3].set_title("(8) find scissors")
     fig.suptitle('Step by step process of scissor detection algorithm', fontsize=20)
     
     for i in range(0, 2):
         for j in range(0, 4):
             ax[i,j].axis("off")
     
     plt.tight_layout(w_pad =3)

def plotNoisedImages(imag, noised_images):
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (10, 8))
    ii = 0
    sigma_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2]
   
    for i in range(0, 3):
        for j in range(0, 2):
        
           
            ax[j, i].imshow(noised_images[ii], cmap=plt.cm.gray)
            #ax[i, j].set_title("Gaussian Noise $\sigma={}$".format(sigma_values[ii]))
            
            #ax[0, 0].set_title("Input image (greyscale)")
           
            ax[j, i].axis("off")
            ax[j, i].axis("off")
            ii += 1
    ax[0, 0].set_title("Input image (greyscale)")
    ax[0, 1].set_title("Gaussian noise $\sigma = 0.5$")
    ax[0, 2].set_title("Gaussian noise $\sigma = 0.8$")
    ax[1, 0].set_title("Gaussian noise $\sigma = 2.0$")
    ax[1, 1].set_title("Gaussian noise $\sigma = 1.5$")
    ax[1, 2].set_title("Gaussian noise $\sigma = 2.0$")
    fig.suptitle('Input images with varying levels of gaussian noise', fontsize=20)
    plt.show()
    
def plotClearBorderImages(imagesBordersRemoved):
    
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (10, 8))
    #clear border plots
    ii = 0
    for i in range(0, 2):
         for j in range(0, 3):
        
            ax[i, j].imshow(imagesBordersRemoved[ii])
            ax[i, j].axis("off")
            ax[i, j].axis("off")
            
            ii += 1
    
    fig.suptitle('Border Objects Removed from Thresholded images of varying noise', fontsize=20)
    plt.tight_layout()
   
def plotRegionLabelledImages(dialated_images, sigma_values):
    
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (10, 8))
    #]label region props plots
    ii = 0
    for i in range(0, 2):
        for j in range(0, 3):
        
            ax[i, j].imshow(dialated_images[ii], cmap = plt.cm.gray)
    
            ax[i, j].set_title("Region labelled image {}'$\sigma = {}$'".format(ii + 1, sigma_values[ii]), fontsize = 12)
            #ax[i, j].set_title("image with gaussian noise (sigma = {})".format(sigma[ii]), fontsize = 14)
    
            ax[i, j].axis("off")
            ax[i, j].axis("off")
            
            ii += 1
    plt.show()
    fig.suptitle('Border Objects Removed from Thresholded images of varying noise', fontsize=20)
    plt.tight_layout()
    
    
def plotIsolatedScissors(small_sciss_list):
    
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (10, 8))
          #function that uses pandas to take tha same information and put it in a csv table
    fig, ax = plt.subplots(ncols = 3, nrows=2, figsize = (10, 8))
    ii = 0
    for i in range(0, 3):
        for j in range(0, 2):
        
            ax[j, i].imshow(small_sciss_list[ii])
            ii += 1
    
    
    plt.tight_layout()

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

def addGaussianNoise(image, sigma, sigma1):
    
    noise = random_noise(image, mode='gaussian',
                         seed=None, clip=True, 
                         var = sigma**2, mean = 0.0)
      
    return noise

def addMedianFilter(image, rank):
    
    
  #  noiseconverted = img_as_ubyte(image, force_copy=False)
    medianFilter = median(image, disk(7))
    return medianFilter

def addMeanFilter(image, rank):
   
    meanFilter = mean(image, disk(7))
    
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
    
def OtsuThresh(noise):
    
    #Generate thresholded image
    threshold = threshold_otsu(noise)
    thresholded_img = (noise > threshold)
        
    #plt.imshow(thresholded_img)
    return threshold, thresholded_img
 
def applyClosingDilation(thresholded_image):
    
    dialated_img = closing(thresholded_image, square(10))
    
    return dialated_img
    
def clearBorder(thresh_image, sigma_values):
    
    edge_touching_removed = clear_border(thresh_image)
   
    return edge_touching_removed

def LabelRegions(image, borders_removed, sigma_values):
    
    #Label connected regions of an integer array using 
    #measure.label Labels each connected entity as one object
    #Connectivity = Maximum number of orthogonal hops to 
    #consider a pixel/voxel as a neighbor. If None, a full 
    #connectivity of input.ndim is used, number of dimensions 
    #of the image. For 2D image it would be 2   
    labelled_image = measure.label(borders_removed)
    lb1 = measure.label(borders_removed)
   
    #Return an RGB image where color-coded labels are painted over the image.
    #Using label2rgb
    image_overlay = label2rgb(labelled_image, image=image, bg_label=0)
 
    return labelled_image, image_overlay, lb1



def calculateRegionProps(label_image, image, sigma_values, ii):

   
    #first we calculate the properties of each region
    #in our labelled image. then we create a table#
    #of all of those properties (area, centroid etc)
    props = regionprops(label_image)
    props1 = regionprops_table(label_image, image,
                               properties=['label','area', 
                               'equivalent_diameter','mean_intensity', 
                               'solidity'])
  
    
    #the unique identifier of the scissors is their euler
    #number is -1. Therefore we can sort the props table
    #by this euler number so that the scissor objects
    #are at the top
    props.sort(key=lambda x: x.area)
    props.sort(key=lambda x: x.euler_number)
     
    #here we loop through the region properties and print
    #the label number, area and euler number
    for i in props:
       print('Label: {}   Object Area: {}   Euler Number: {} '
             .format(i.label, i.area, i.euler_number))
    

    #the next thing we do is to apply a mask. this will
    #allow use to only show the smallest scissor object
    #we loop over all of the sigma values and apply the
    #mask
    scissor_mask = label_image
        
    #basically if the length o the props is > 1 then
    #we know we have a non black pixel
    if len(props)>1:
            #1 is calling forst row, rg index used in the loop
            #since we sorted our props table above we can set 
            #everything except the row dentoted by props[1:] to bloack
            #since the smallest scissors will be in the first row
            #then it will be the only object that does not get set
            #to black
        for tablerow in props[1:]:
                # find everything in table that is in first row 
                #and makes it black/ sets it to 0
            scissor_mask[tablerow.coords[:,0], 
                             tablerow.coords[:,1]] = 0
    #inverting the mask to let it not equal 0/ black. 
    #Everyting else becomes 0
    scissor_mask[scissor_mask!=0] = 1
    #then we asign our smallest scissor t this mask
    #which is the image with the detected smaller scissor
    smallest_scissors = scissor_mask

   

    #in order to get the smallest area props we can set
    #a few vars below
    num_scissors = 0
    scissor_centroids = []
    scissor_areas = []
     
    #here we loop through the region labbled image's props and
    #use the euler number to calculate the area etc
    for props in props:  

         #to detect objects with two holes set euler number == -1
         if props.euler_number == -1:
                
        #if scissor has been detected append and append area and 
        #centroid
                    num_scissors += 1
                    scissor_centroids.append(props.centroid)
                    scissor_areas.append(props.area)
                     
     #now i get the min value in the list to get smallers area 
     #and also  the index of the min value in order to use it 
     #to get the equvelent centroid in the other list by index
    area_of_smaller_scissors = min(scissor_areas)
        
     #get index of smallerst area as the smallest centroid will be in
     #the same index in its list
    centroid_index = scissor_areas.index(area_of_smaller_scissors)
    centroid_of_scissor = scissor_centroids[centroid_index]
     
    
    #print the output to the screen
    print("\nthe num of scissors is: ",
          num_scissors)
    print("\nthe smaller scissors has an area of: ",
          area_of_smaller_scissors)
    print("\nthe centroid of the smaller scissors is: ",
          centroid_of_scissor)
    print("centroid index is ", centroid_index)
    
    

     #using skimage.draw we can use the centroid infromation
     # to draw the cordinates of the smallest scissor centroid 
     #on the image directly. #here we calculate the x and y coords 
     #of each line using the centroid info
    x_coord1, y_coord1 = line(0, int(centroid_of_scissor[1]),
                        smallest_scissors.shape[0] - 1, 
                        int(centroid_of_scissor[1]))
    
    x_coord2, y_coord2 = line(int(centroid_of_scissor[0]), 
                        0, int(centroid_of_scissor[0]), 
                        smallest_scissors.shape[1] - 1)
    
    #this line just changes the opacitiy if you will of the 
    #line (mask), range(1, 255) use one for best results
    smallest_scissors[x_coord1, y_coord1] = 1
    smallest_scissors[x_coord2, y_coord2] = 1
     
    return smallest_scissors, props1
    



 
main()
