from matplotlib import pyplot as plt
import cv2
import os
import random

root_directory = "/Users/gouravchirkhare/PycharmProjects/Unet_segment/Data/"

Filtered_img_dir = root_directory + '256_patches/Filtered_mask_and_images/F_images/'
Filtered_mask_dir = root_directory + '256_patches/Filtered_mask_and_images/F_masks/'

Filtered_img_list = os.listdir(Filtered_img_dir)
Filtered_msk_list = os.listdir(Filtered_mask_dir)

num_images = len(os.listdir(Filtered_img_dir))
img_num = random.randint(0, num_images - 1)

img_for_plot = cv2.imread(Filtered_img_dir + Filtered_img_list[img_num], 1)
img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

mask_for_plot = cv2.imread(Filtered_mask_dir + Filtered_msk_list[img_num], 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()