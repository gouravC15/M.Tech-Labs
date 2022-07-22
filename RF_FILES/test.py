import numpy as np
import cv2
import pandas as pd
import time

start_main = time.time()
def feature_extraction(img):
    df = pd.DataFrame()
    print("\n[2/4] GENERATING FEATURES..")
    start = time.time()
    # reshape img into single column
    img2 = img.reshape(-1)
    df = pd.DataFrame()
    df['Original Image'] = img2

    # Generate Gabor features
    num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    print("[3/4] Processing: [",end=" ")
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
                    df[gabor_label] = filtered_img.astype('int16')  # Labels columns as Gabor1, Gabor2, etc.
                    # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    print(num, end=" ")
                    num += 1  # Increment for gabor column label

    # Gerate OTHER FEATURES and add them to the data frame
    # CANNY EDGE
    edges = cv2.Canny(img, 100, 200)  # Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1.astype('int16')  # Add column to original dataframe

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
    df['Gaussian s3'] = gaussian_img1.astype('int16')

    # GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3.astype('int16')

    # MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1.astype('int16')
    end = time.time()
    feature_generation_time = end - start
    print("....100%]\n      Time: {} min".format(round(feature_generation_time, 1)/60),"\n      [+] GENERATED")
    return df

import joblib
from matplotlib import pyplot as plt

PATH ='/Users/gouravchirkhare/PycharmProjects/Unet_segment/R-Forest/'

print("\n[1/4] LOADING MODEL..")
filename = PATH+"Saved_Model/Model2_int16.sav"
loaded_model = joblib.load(filename)
print("      [+] MODEL LOADED")

#input_img = cv2.imread(PATH+"RF_DATA/02Train_image.tif")
input_img = cv2.imread(PATH+"RF_DATA/Test_image.tif")
#input_img = cv2.imread("/Users/gouravchirkhare/Documents/CS Mtech/Project_Data/TrueColor.png")
img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

#extracting features
X = feature_extraction(img)
print("\n[4/4] PREDICTING..")
result = loaded_model.predict(X)
segmented = result.reshape((img.shape))
print("      [*]ALL DONE ")
end_main = time.time()
complete_time = end_main - start_main
print("--- Total Time: {} min ---".format(round(complete_time)/60))

plt.title('Testing Image')
plt.imshow(input_img)
plt.title('Segmented Image')
plt.imshow(segmented)
plt.show()