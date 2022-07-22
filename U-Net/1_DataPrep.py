import numpy as np
#from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import cv2
import os
'''
#Quick understanding of the dataset
temp_img = cv2.imread("Data/Origial/images/M-33-7-A-d-2-3.tif") #3 channels / spectral bands
plt.imshow(temp_img[:,:,2]) #View each channel...
plt.show()
temp_mask = cv2.imread("Data/Origial/masks/M-33-7-A-d-2-3.tif") #3 channels but all same.
labels, count = np.unique(temp_mask[:,:,0], return_counts=True) #Check for each channel. All chanels are identical
print("Labels are: ", labels, " and the counts are: ", count)
'''
root_directory = "/Users/gouravchirkhare/PycharmProjects/Unet_segment/Data/"
patch_size = 256
# divide all images into patches of 256x256x3.
# Read images from repsective 'images' subdirectory
# As all images are of different size we have 2 options, either resize or crop
# But, some images are too large and some small. Resizing will change the size of real objects.
# Therefore, we will crop them to a nearest size divisible by 256 and then
# divide all images into patches of 256x256x3.
counter_i = 0
img_dir=root_directory+"Original/images"
for path, subdirs, files in os.walk(img_dir):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    # print(dirname)
    images = os.listdir(path)  # List of all image names in this subdirectory
    # print(images)
    for i, image_name in enumerate(images):
        if image_name.endswith(".tif"):
            # print(image_name)
            image = cv2.imread(path+"/"+image_name, 1)  # Read each image as BGR
            SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
            SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
            image = Image.fromarray(image)
            image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            image = np.array(image)
            # Extract patches from each image
            #print("Now patchifying image:", path+"/"+image_name)
            patches_img = patchify(image, (256, 256, 3), step=256)  # Step=256 for 256 patches means no overlap
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :]
                    # single_patch_img = (single_patch_img.astype('float32')) / 255. #We will preprocess using one of the backbones
                    single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
                    cv2.imwrite(root_directory+"256_patches/P_images/"+image_name+"patch_"+str(i)+str(j)+".tif", single_patch_img)
                    #print("Stored: ","Data/Origial/256_patches/P_images/:"+image_name+"_patch_",str(i)+str(j)+".tif")
                    counter_i += 1
print("Image Patching Complete [affected]: ",counter_i)

counter_i=0
mask_dir=root_directory+"Original/masks/"
for path, subdirs, files in os.walk(mask_dir):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    masks = os.listdir(path)  # List of all image names in this subdirectory
    for i, mask_name in enumerate(masks):
        if mask_name.endswith(".tif"):
            mask = cv2.imread(path+"/"+mask_name,0)  # Read each image as Grey (or color but remember to map each color to an integer)
            SIZE_X = (mask.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
            SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
            mask = Image.fromarray(mask)
            mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            mask = np.array(mask)
            # Extract patches from each image
            #print("Now patchifying mask:",path+"/"+mask_name)
            patches_mask = patchify(mask, (256, 256), step=256)  # Step=256 for 256 patches means no overlap
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i, j, :, :]
                    # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                    # single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
                    cv2.imwrite(root_directory+"256_patches/P_masks/"+mask_name+"patch_"+str(i)+str(j)+".tif", single_patch_mask)
                    #print("Stored: ", "Data/Origial/256_patches/P_masks/:" + mask_name + "_patch_",str(i) + str(j) + ".tif")
                    counter_i += 1
print("Masks Patching Complete [affected]: ", counter_i)