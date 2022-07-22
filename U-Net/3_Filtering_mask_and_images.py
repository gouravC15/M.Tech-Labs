import cv2
import os
import numpy as np
from threading import Thread
from time import *

root_directory = "/Users/gouravchirkhare/PycharmProjects/Unet_segment/Data/"

# list of patched img & masks
train_img_dir = root_directory + "256_patches/P_images/"
train_mask_dir = root_directory + "256_patches/P_masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)
# filtering and saving only useful images

useless = 0  # Useless image counter

def filter(img_list,msk_list,useless):
    for img in range(len(img_list)):  # Using t1_list as all lists are of same size
        img_name = img_list[img]
        mask_name = msk_list[img]
        print("Now preparing image and masks number: ", img)
        temp_image = cv2.imread(train_img_dir + img_list[img], 1)
        temp_mask = cv2.imread(train_mask_dir + msk_list[img], 0)
        # temp_mask=temp_mask.astype(np.uint8)

        val, counts = np.unique(temp_mask, return_counts=True)

        if (1 - (counts[0] / counts.sum())) > 0.05:  # At least 5% useful area with labels that are not 0
            # print(":Valid")
            cv2.imwrite(root_directory + '256_patches/Filtered_mask_and_images/F_images/' + img_name, temp_image)
            cv2.imwrite(root_directory + '256_patches/Filtered_mask_and_images/F_masks/' + mask_name, temp_mask)
        else:
            # print(":Invalid")
            useless += 1

filter_obj = Thread(target=filter,args=(img_list,msk_list,useless))

start = time()
filter_obj.start()
end = time()

print("Total useful images are: ", len(img_list) - useless)  # 20,075
print("Total useless images are: ", useless)  # 21,571

print("(MultiThread Done)\n")

if __name__ == "__main__":
	sleep(0.1)
	print("\nExecution time for Filtering is: ", (end-0.1) - start)