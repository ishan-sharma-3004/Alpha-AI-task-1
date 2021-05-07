from skimage.restoration import denoise_nl_means,estimate_sigma  #for local mean
from skimage import data,img_as_float,img_as_ubyte,io   #for importing image nd conversion to 8it
import numpy as np                                      #for array operations    
import cv2                                              #for opencv library
from matplotlib import pyplot as plt                    #plotting functions

img=img_as_float(io.imread('/Users/ishansharma/Desktop/Screenshot 2021-05-05 at 4.20.26 PM.jpeg'))
sigma_est=np.mean(estimate_sigma(img,multichannel=True))                                             #finding mean of the image(in 2d array)
denoise=denoise_nl_means(img,h=1.15*sigma_est,fast_mode=True,patch_size=5,patch_distance=6000,multichannel=True)     #performing loacl means on image
denoise_ubyte= img_as_ubyte(denoise)       #converting image to 8bit format
plt.imshow(denoise, cmap='gray')
plt.hist(denoise_ubyte.flat,bins=100,range=(0,255))        #getting hostogram of the image in the range 0,255
segm1=(denoise_ubyte <= 55)                                #segmenting images into 4 segments according to the values mentioned
segm2=(denoise_ubyte > 55) & (denoise_ubyte <= 110)
segm3=(denoise_ubyte > 110) & (denoise_ubyte <= 210)
segm4=(denoise_ubyte > 210)
all_segments=np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))#creating a similar array of 0's of the size of the image
all_segments[segm1]=(1,0,0)            ##separating all segments using different colors to distinguish between them
all_segments[segm2]=(0,1,0)
all_segments[segm3]=(0,0,1)
all_segments[segm4]=(1,1,0)

from scipy import ndimage as nd
segm1_opened=nd.binary_opening(segm1,np.ones((3,3)))             #binary opening and closing to take care of stray pixels and voids in the image      
segm1_closed=nd.binary_closing(segm1_opened,np.ones((3,3)))

segm2_opened=nd.binary_opening(segm2,np.ones((3,3)))
segm2_closed=nd.binary_closing(segm2_opened,np.ones((3,3)))

segm3_opened=nd.binary_opening(segm3,np.ones((3,3)))
segm3_closed=nd.binary_closing(segm3_opened,np.ones((3,3)))

segm4_opened=nd.binary_opening(segm4,np.ones((3,3)))
segm4_closed=nd.binary_closing(segm4_opened,np.ones((3,3)))

all_segments_cleaned=np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))          #just like above similar array of 0's of same size as image 
all_segments_cleaned[segm1_closed]=(1,0,0)                 #putting colors to the segments to distinguish between  
all_segments_cleaned[segm2_closed]=(0,1,0)      
all_segments_cleaned[segm3_closed]=(0,0,1)
all_segments_cleaned[segm4_closed]=(1,1,0)

plt.imshow(all_segments_cleaned)           #final output
plt.imsave('/Desktop/segmented.jpeg',all_segments_cleaned)

