import os
from keras.models import load_model
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob

if __name__ == "__main__":
    model = load_model("Data/Model/landcover_25_epochs_RESNET_backbone_batch16.hdf5", compile=False)
    img_path = "Data/Train_Test_Ready/Data_for_keras/Test/i1.tif"
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    original_img = img

    h,w,_ = img.shape

    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)

    predict_mask = model.predict(img)
    predict_mask = predict_mask[0]

    predict_mask=np.concatenate(
        [predict_mask, predict_mask, predict_mask], axis=2
    )

    predict_mask = (predict_mask>0.5)*255
    predict_mask = predict_mask.astype(np.float32)
    predict_mask = cv2.resize(predict_mask,(w,h))

    original_img = original_img.astype(np.float32)

    alpha =0.6
    cv2.addWeighted(predict_mask,alpha,original_img,1-alpha,0,original_img)

    cv2.imshow(original_img)