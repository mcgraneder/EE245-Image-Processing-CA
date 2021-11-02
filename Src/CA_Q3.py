from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms


def main():
    
     #This function takes in an image reads it in and retursn the image
    img1, img2 = readInImage('Snapchat-190545990.jpg', "reference.jpg")
    
    #function that takes in the resized image and converts it
    #to greyscale
    resized_image1, resized_image2 = resizeImage(img1, img2)
    
    matched = addMatchHostogramToImage(resized_image1, resized_image2)
    
    #generate a plot showing the resized input image, the
    #referance image and the combined colour mathed output image
    plotsForPartA(resized_image1, resized_image2, matched)
    
    #generate a plot showing the respective histogram plots
    #for each R,G,B channel for each image (input, ref, matched imges)
    addHistogramPlots(resized_image1, resized_image2, matched)
    
    #optional for extra explaintion of histograms in report
    addCombinedHistogramPlots(resized_image1, resized_image2, matched)

    
    
#define a function that reads in an image and 
#returns the image for use in the programme
def readInImage(image1, image2):
    
    img1 = io.imread(image1)
    img2 = io.imread(image2)
    return img1, img2


#define a function that takes in an image and both
#resizes it and converts it to greyscale
def resizeImage(image1, image2):
    
    img_resized1 = resize(image1, (512, 512))
    img_resized2 = resize(image2, (512, 512))
    return img_resized1, img_resized2


#define a functgion to add histogram colour matching to our
#original resized source image
def addMatchHostogramToImage(source_image, referance_image):
    
    matched = match_histograms(source_image, referance_image, multichannel=True)
    
    return matched


#define function to plot the desired results for part a
def plotsForPartA(original_image, referance, matched_image):
    
    #next we plot
    fig, ax = plt.subplots(ncols = 3, figsize = (24, 6))

    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3)


    ax[0].imshow(original_image, cmap=plt.cm.gray)
    ax[0].set_title("original source image (512 X 512)", fontsize = 25)
    ax[0].axis("off")

    ax[1].imshow(referance, cmap=plt.cm.gray)
    ax[1].set_title("referance (resized 512 x 512)",fontsize = 25)
    ax[1].axis("off")
    
    ax[2].imshow(matched_image, cmap=plt.cm.gray)
    ax[2].set_title("matched_image (512 X 512)" ,fontsize = 25)
    ax[2].axis("off")
    
    #may have to create your own folder 
    #plt.tight_layout()
    fig.suptitle('plot of source, referance and colour matched images', fontsize=35)
    plt.show()
    
  
    
#define a function which adds hisogram and cumulative histogram
#plots for each rgb channel to each of our inages, source, ref and matched
def addHistogramPlots(resized_image1, resized_image2, matched):
    
    fig, ax = plt.subplots(ncols = 3, nrows = 4, figsize = (20, 20))
    arr = [resized_image1, resized_image2, matched]
    
    ax[0, 2].axis("off")
    
    for i in range(0, 3):
        for j in range(0, len(arr)):
            img_hist, bins = exposure.histogram(arr[i][..., j], source_range='dtype')
            img_cdf, bins = exposure.cumulative_distribution(arr[i][..., j])
            
            colourArray = ["r", "g", "b"]
            if (i == 0):
                ax[i, j].imshow(arr[j], cmap=plt.cm.gray)
                ax[i, j].axis("off")
                
            ax[j + 1, i].plot(bins, img_hist / img_hist.max(), label="histogram", color = colourArray[j])
            ax[j + 1, i].plot(bins, img_cdf, label="cumulative histogram", color = "darkorange", linewidth = 2, linestyle = "dashed")
            ax[j + 1, i].set_label("test")
            ax[j + 1, i].legend(loc ="upper left", prop={'size': 15})
            ax[j + 1, i].set_xlabel(" ",fontsize = 25) 
           
       
        
       
    ax[0, 0].set_title("source",fontsize = 35) 
    ax[0, 1].set_title("referance",fontsize = 35) 
    ax[0, 2].set_title("matched",fontsize = 35)
    ax[1, 0].set_title("source histogram plots",fontsize = 25) 
    ax[1, 1].set_title("referance histogram plots",fontsize = 25) 
    ax[1, 2].set_title("matched histogram plots",fontsize = 25) 
    ax[1, 0].set_ylabel("red",fontsize = 15) 
    ax[2, 0].set_ylabel("green",fontsize = 15) 
    ax[3, 0].set_ylabel("blue",fontsize = 15)
    
    plt.tight_layout(w_pad = 8.0, h_pad = 5.0)
    
    plt.show()
    
def addCombinedHistogramPlots(resized_image1, resized_image2, matched): 
    
     fig, ax = plt.subplots(ncols = 3, nrows = 2, figsize = (20, 12))
    
     arr = [resized_image1, resized_image2, matched]
     print(len(arr))
     for i in range(0, 1):
        for j in range(0, len(arr)):
           
            
            
            if (i == 0):
                ax[i, j].imshow(arr[j], cmap=plt.cm.gray)
                ax[i, j].axis("off")
            
        
            ax[i + 1, j].hist(arr[j][:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
            ax[i + 1, j].hist(arr[j][:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
            ax[i + 1, j].hist(arr[j][:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
            ax[i + 1, j].set_xlabel('Intensity Value', fontsize = 19)
            ax[i + 1, j].set_ylabel('Count', fontsize = 19)
            ax[i + 1, j].legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'], prop={'size': 12})   
            
           
     ax[0, 0].set_title("source",fontsize = 25) 
     ax[0, 1].set_title("referance",fontsize = 25) 
     ax[0, 2].set_title("matched",fontsize = 25) 
     ax[1, 0].set_title("source combined histogram",fontsize = 20) 
     ax[1, 1].set_title("referance combined histogram",fontsize = 20) 
     ax[1, 2].set_title("matched combined histogram",fontsize = 20) 
     
     plt.tight_layout(pad = 4.0)  
     #plt.savefig("CA1_out/Q3/Q3_part2b.jpg")
     fig.suptitle('histogram plots of source, referance and matched images', fontsize=25)
     plt.show()
    
#then we call main to run the programme   
main()


