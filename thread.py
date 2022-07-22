import cv2
from skimage.filters.rank import entropy,gradient
from skimage.morphology import disk
from matplotlib import pyplot as plt
from skimage.filters import roberts, scharr, prewitt, sobel,rank,
from scipy import ndimage as nd
import numpy as np

PATH = '/Users/gouravchirkhare/PycharmProjects/Unet_segment/R-Forest/'
save = '/Users/gouravchirkhare/Desktop/Filters/'
img1 = cv2.imread(PATH + "RF_DATA/02Train_image.tif")
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)





'''
gaussian_img = nd.gaussian_filter(img, sigma=3)
cv2.imshow('gaussian_img',gaussian_img)

median_img = nd.median_filter(img, size=6)
cv2.imshow('median_img',median_img)

sobel_img = sobel(img)
cv2.imshow('sobel_img',sobel_img)

entropy_img = entropy(img, disk(1))
cv2.imshow('entropy_img',entropy_img)

edge_roberts = roberts(img)
cv2.imshow('edge_roberts',edge_roberts)

edge_scharr = scharr(img)
cv2.imshow('edge_scharr',edge_scharr)
'''
gradient_img = gradient(img,disk(5))
cv2.imshow('gradient_img',gradient_img)

edge_prewitt = prewitt(img)
cv2.imshow('edge_prewitt',edge_prewitt)



otsu_img = rank.otsu(img,disk(10))
cv2.imshow('otsu_img',otsu_img)


equilizer = rank.equalize(img, disk(10))
cv2.imshow('equilizer',equilizer)

cv2.waitKey(0)
cv2.destroyAllwindows()

'''
gaussian_img = nd.gaussian_filter(img, sigma=3)
cv2.imwrite(save+'gaussian_img.jpg',gaussian_img)
median_img = nd.median_filter(img, size=3)
cv2.imwrite(save+'median_img.jpg',median_img)

sobel_img = sobel(img)
cv2.imwrite(save+'sobel_img.jpg', sobel_img)
entropy_img = entropy(img, disk(1))
cv2.imwrite(save+'entropy_img.jpg', entropy_img)
edge_roberts = roberts(img) #useless
cv2.imwrite(save+'edge_roberts.jpg',edge_roberts)
edge_scharr = scharr(img)
cv2.imwrite(save+'edge_scharr.jpg',edge_scharr)
edge_prewitt = prewitt(img)
cv2.imwrite(save+'edge_prewitt.jpg',edge_prewitt)
farid_img=farid(img)
cv2.imwrite(save+'farid_img.jpg',farid_img)
'''