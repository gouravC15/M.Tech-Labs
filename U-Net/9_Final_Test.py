from keras.models import load_model
from tensorflow.python.keras.metrics import MeanIoU
from matplotlib import pyplot as plt
import numpy as np
import random

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
import segmentation_models as sm

scaler = MinMaxScaler()
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

model = load_model("/Users/gouravchirkhare/PycharmProjects/Unet_segment/Data/Model/landcover_25_epochs_RESNET_backbone_batch16.hdf5", compile=False)
print("[Task]: Testing \n[Model]:"+BACKBONE+"\n[Data]: Multispectral imagery")

#batch_size=32 #Check IoU for a batch of images

#Test generator using validation data.
##addition
seed = 24
batch_size = 8
n_classes = 4

def preprocess_data(img, mask, num_class):
    # Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)  # Preprocess based on the pretrained backbone...
    # Convert mask to one-hot
    mask = to_categorical(mask, num_class)
    return (img, mask)

def trainGenerator(train_img_path, train_mask_path, num_class):
    img_data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        batch_size=batch_size,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)
    print("Pre-processing Complete.")

print("[Val] TrainGenerator Started..")
val_img_path = "/Users/gouravchirkhare/PycharmProjects/Unet_segment/Data/Train_Test_Ready/Data_for_keras/val_images/"
val_mask_path = "/Users/gouravchirkhare/PycharmProjects/Unet_segment/Data/Train_Test_Ready/Data_for_keras/val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=4)
print("[Val] TrainGenerator Complete.")
##
test_image_batch, test_mask_batch = val_img_gen.__next__()

#Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3)
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#######################################################
#View a few images, masks and corresponding predictions.
img_num = random.randint(0, test_image_batch.shape[0]-1)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image_batch[img_num])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_batch_argmax[img_num])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_pred_batch_argmax[img_num])

#colorbar
orig_map=plt.cm.get_cmap('viridis')
# reversing the original colormap using reversed() function
reversed_map = orig_map.reversed()
print("\n\n")
plt.colorbar(label="Vegetation", orientation="horizontal")
plt.show()