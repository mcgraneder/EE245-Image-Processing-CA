from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.transform import resize
from skimage.filters.rank import median, mean
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

import timeit
def main():
    
    start = timeit.default_timer()
    #This function takes in an image reads it in and retursn the image
    img = readInImage('Snapchat-190545990.jpg')
    
    #function that takes in the read in image and resizes it
    resized_image = resizeImage(img)
    
    #function that takes in the resized image and converts it
    #to greyscale
    grayscale = convertImageToGreyscale(resized_image)
    
    #function that takes in the greyscale image and finds the
    #mean, min and max greyscale intensity values (prints to console)
    findGreyscaleValues((grayscale))
    
    #function that takes in the greyscale image and adds gaussian noise
    noise = addGaussianNoise(grayscale, 0.1)
    
    #function that takes the noised image and applys a 10 x 10
    #median rank filter
    medianFilter = addMedianFilter(noise, 10)
    
    #function that takes the noised image and applys a 10 x 10
    #mean rank filter
    meanFilter = addMeanFilter(noise, 10)#
    
    
    standard_deviations = [0.5, 1, 1.5, 2, 2.5, 3]
    #out put array which stores processd images with gaussian
    #noise reduction
    noise_reduced_images = []
    for i in range(len(standard_deviations)):
        
        #function that takes in an array of different standard deviation values
        #and applies gaussian noise reduction for each value. 
        filtered = addGaussianNoiseReduction(noise, standard_deviations[i])
        
        #the output of this function gets appened to an output array for plotting
        noise_reduced_images.append(filtered)
    
    
    #this block of code calls 4 seperate functions which
    #render seperate plots for each part of q1. and save them
    #to and out folder
    plotsForPartA(resized_image, grayscale)
    plotsForPartB(grayscale, noise)
    plotsForPartC(noise, medianFilter, meanFilter)
    plotsForPartD(noise, noise_reduced_images, standard_deviations)
    
   



    stop = timeit.default_timer()

    print('Time: ', stop - start)  

#define a function that reads in an image and 
#returns the image for use in the programme
def readInImage(image):
    
    img = io.imread(image)
    return img
  
#define a function that takes in an image and both
#resizes it and converts it to greyscale
def resizeImage(image):
    
    img_resized = resize(image, (512, 512))
    return img_resized

#define a function that takes in an image and
#converts it to greyscale
def convertImageToGreyscale(image):
    
    grayscale = rgb2gray(image)
    return grayscale

#define a function that takes in an image and
#converts it to greyscale
def findGreyscaleValues(image):
    
    #print out the shape and max/min values of the grayscale
    print("shape: ", image.shape)
    print("Min/Max/mean values: ", image.min(), image.max(), image.mean())

#function that adds noise to the grayscale image
def addGaussianNoise(image, sigma):
    
    noise = random_noise(image, mode='gaussian', seed=None, clip=True, var = sigma**2, mean = 0.0)
    return noise

#function that adds a median rank filer to the noisy image
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

#function that removes the noise using gaussian noise reduction
def addGaussianNoiseReduction(image, standard_deviation):
    
    #standard_deviation = 1.6
    filtered = gaussian(image, sigma = standard_deviation, mode="reflect")
    return filtered

#define function to plot the desired results for part a
def plotsForPartA(original_image_resized, grayscale_image):
    
    #next we plot
    fig, ax = plt.subplots(ncols = 2, figsize = (14, 7))

   


    ax[0].imshow(original_image_resized, cmap=plt.cm.gray)
    ax[0].set_title("original resized image", fontsize = 17)
    ax[0].axis("off")

    ax[1].imshow(grayscale_image, cmap=plt.cm.gray)
    ax[1].set_title("greyscale image (512 x 512)",fontsize = 17)
    ax[1].axis("off")
    fig.suptitle('Rezied Image and greyscale plot', fontsize=23)
    #may have to create your own folder 
    #plt.savefig("CA1_out/Q1/Q1_part1.jpg")
    #plt.tight_layout()
    
    plt.show()
    
def plotsForPartB(grayscale_image, image_with_gaussian_noise):
    
    #next we plot
    fig, ax = plt.subplots(ncols = 2, figsize = (14, 7))

   


    ax[0].imshow(grayscale_image, cmap=plt.cm.gray)
    ax[0].set_title("Greyscale image", fontsize = 17)
    ax[0].axis("off")

    ax[1].imshow(image_with_gaussian_noise, cmap=plt.cm.gray)
    ax[1].set_title("Image with gaussian noise $var = 0.1$",fontsize = 17)
    ax[1].axis("off")
    fig.suptitle('Comparison of greyscale image with gaussian noise', fontsize=23)
    #may have to create your own folder 
    #plt.savefig("CA1_out/Q1/Q1_part1.jpg")
    #plt.tight_layout()
    
    plt.show()

def plotsForPartC(image_with_gaussian_noise, image_with_median_rank, image_with_mean_rank):
    
    #next we plot
    fig, ax = plt.subplots(ncols = 3, figsize = (24, 9))

  


    ax[0].imshow(image_with_gaussian_noise, cmap=plt.cm.gray)
    ax[0].set_title("noisy image", fontsize = 30)
    ax[0].axis("off")

    ax[1].imshow(image_with_median_rank, cmap=plt.cm.gray)
    ax[1].set_title("median rank filter (10 x 10)",fontsize = 30)
    ax[1].axis("off")
    
    ax[2].imshow(image_with_mean_rank, cmap=plt.cm.gray)
    ax[2].set_title("mean rank filter (10 x 10)",fontsize = 30)
    ax[2].axis("off")
    fig.suptitle('Comparison of mean & median spatial filters\n', fontsize=38)
    #may have to create your own folder 
    plt.savefig("CA1_out/Q1/Q1_part3.jpg")
    plt.tight_layout(w_pad=3)
    
    plt.show()

def plotsForPartD(noised_image, gaussian_noise_reduced_images, standard_devs):
    
    fig, ax = plt.subplots(ncols = 3, nrows = 2, figsize = (12, 10))
    
    ii = 0
    
        
    for i in range(2):
        for j in range(3):
            
            if(i == 0 and j == 0):
                ax[i][j].imshow(noised_image, cmap = plt.cm.gray)
            else:
                ax[i][j].imshow(gaussian_noise_reduced_images[ii], 
                                cmap = plt.cm.gray)
                ax[i][j].set_title("Gaussian noise reduction\n $\sigma = {}$".format(standard_devs[ii]), fontsize = 20)
                
            ax[i][j].axis("off")
            ii += 1
            
    ax[0][0].set_title("Original noised image", fontsize = 18)    
    fig.suptitle('Applied Gaussian noise reduction for varying standard deviation', fontsize=25)
    plt.tight_layout()
    plt.show()
    
#then we call main to run the programme   
main()  

