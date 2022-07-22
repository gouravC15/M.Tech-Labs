#After filtering images and masks this is to split data into Train and Test
#Images will be takin from Ganesh.c/Filtered_mask_and_images/
# & will be stored in Train_Test_Ready
import splitfolders
import os

root_directory = "/Users/gouravchirkhare/PycharmProjects/Unet_segment/Data/"

input_folder = root_directory+'256_patches/Filtered_mask_and_images/'
output_folder = root_directory+'Train_Test_Ready/temp'
#it  will create 2 dir in o/p folde 1.test, 2.val
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
dir = os.listdir(output_folder)         # Getting the list of directories

if len(dir) == 0:               # Checking if the o/p list is empty or not
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25),group_prefix=None)  # default values
    print("Splitting Successful")
else:
    print("Directory: "+output_folder+"\nNot Empty. \tFirst clear it to continue..")

#NEXT>> dir struct for training & test
# Now manually move folders around to bring them to the following structure.
"""
Your current directory structure:
Data/
    train/
        images/
            img1, img2, ...
        masks/
            msk1, msk2, ....
    val/
        images/
            img1, img2, ...
        masks/
            msk1, msk2, ....

Copy the folders around to the following structure... 
Data/
    train_images/
                train/
                    img1, img2, img3, ......

    train_masks/
                train/
                    msk1, msk, msk3, ......

    val_images/
                val/
                    img1, img2, img3, ......                
    val_masks/
                val/
                    msk1, msk, msk3, ......
"""
#Next>> Training
# install>> pip3 install -U segmentation-models