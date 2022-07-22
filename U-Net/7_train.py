import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
# Define Generator for images and masks so we can read them directly from the drive.

seed = 24
batch_size = 16
n_classes = 4

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#from keras.utils import to_categorical(replace with below)
from keras.utils.np_utils import to_categorical

# Use this to preprocess input for transfer learning
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

print("Pre-processing Started..")
# Define a function to perform additional preprocessing after datagen.
# For example, scale images, convert masks to categorical, etc.
def preprocess_data(img, mask, num_class):
    # Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)  # Preprocess based on the pretrained backbone...
    # Convert mask to one-hot
    mask = to_categorical(mask, num_class)
    return (img, mask)


# Define the generator.
# We are not doing any rotation or zoom to make sure mask values are not interpolated.
# It is important to keep pixel values in mask as 0, 1, 2, 3, .....

#from tensorflow.keras.preprocessing.image import ImageDataGenerator(insted this use below)
from keras.preprocessing.image import ImageDataGenerator

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

print("[Train] TrainGenerator Started..")
train_img_path = "Data/Train_Test_Ready/Data_for_keras/train_images/"
train_mask_path = "Data/Train_Test_Ready/Data_for_keras/train_masks/"
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=4)
print("[Train] TrainGenerator Complete.")

print("[Val] TrainGenerator Started..")
val_img_path = "Data/Train_Test_Ready/Data_for_keras/val_images/"
val_mask_path = "Data/Train_Test_Ready/Data_for_keras/val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=4)
print("[Val] TrainGenerator Complete.")

#Make sure the generator is working and that images and masks are indeed lined up.
#Verify generator.... In python 3 next() is renamed as __next__()

x, y = train_img_gen.__next__()
#x=image, y=mask //mask will be converted into categorical
for i in range(0,3):
    image = x[i]
    mask = np.argmax(y[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()

x_val, y_val = val_img_gen.__next__()

for i in range(0,3):
    image = x_val[i]
    mask = np.argmax(y_val[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()

###########################################################################
#Define the model metrcis and load model.

num_train_imgs = len(os.listdir('Data/Train_Test_Ready/Data_for_keras/train_images/train/'))
num_val_images = len(os.listdir('Data/Train_Test_Ready/Data_for_keras/val_images/val/'))
steps_per_epoch = num_train_imgs//batch_size
val_steps_per_epoch = num_val_images//batch_size


IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]

n_classes=4

#############################################################################
#Use transfer learning using pretrained encoder in the U-Net
#(make sure you uncomment the preprocess_input part in the
# preprocess_data function above)
################################################################
#Define the model
# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet',input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),classes=n_classes, activation='softmax')
model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])

#Other losses to try: categorical_focal_dice_loss, cce_jaccard_loss, cce_dice_loss, categorical_focal_loss

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
print(model.summary())
print(model.input_shape)
#Fit the model
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
#Train the model.
print("Training Started..")
history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=25,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch)
model.save('Model/landcover_25_epochs_RESNET_backbone_batch16.hdf5')
print("Training Complete.")

##################################################################
#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()