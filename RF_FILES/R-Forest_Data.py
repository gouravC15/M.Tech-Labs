import numpy as np
import cv2
import pandas as pd
import time
import os

PATH ='/Users/gouravchirkhare/PycharmProjects/Unet_segment/R-Forest/'

img = cv2.imread(PATH+"RF_DATA/02Train_image.tif")
print("Original Img type: ", img.dtype, " Size: ", img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Gray Img type: ", img.dtype, " Size: ", img.shape)

# reshape img into single column
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2.astype('int8')

# Generate Gabor features
num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(2):  # Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  # Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
            for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5

                gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                #                print(gabor_label)
                ksize = 9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                # Now filter the image and add values to a new column
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img.astype('int8')  # Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  # Increment for gabor column label

########################################
print(df.head())
# Gerate OTHER FEATURES and add them to the data frame

# CANNY EDGE
edges = cv2.Canny(img, 100, 200)  # Image, min and max values
edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1.astype('int8')  # Add column to original dataframe

from skimage.filters import roberts, sobel, scharr, prewitt

# ROBERTS EDGE
edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1.astype('float16')

# SOBEL
edge_sobel = sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1.astype('float16')

# SCHARR
edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1.astype('float16')

# PREWITT
edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1.astype('float16')

# GAUSSIAN with sigma=3
from scipy import ndimage as nd

gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1.astype('int8')

# GAUSSIAN with sigma=7
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3.astype('int8')

# MEDIAN with sigma=3
median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1.astype('int8')

# Telling GROUND TRUTH
labeled_img = cv2.imread(PATH+"RF_DATA/02Train_mask2.tif")
labeled_img_show = labeled_img
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1
print(df.head())

if not os.path.exists(PATH+"RF_DATA/Features_8bit.csv"):
    print("\n[..] CREATING: Features_8bit.csv")
    start = time.time()
    # df.to_csv("./Features/Features.csv")
    df.to_csv(PATH+"RF_DATA/Features_8bit.csv")
    end = time.time()
    modin_duration = end - start
    print("Time to read with Modin: {} seconds".format(round(modin_duration, 1)))
    print("\n[+] CREATED: Features_8bit.csv")
else:
    print("\n[+] EXISTS: Features_8bit.csv")