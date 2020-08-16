
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils
from imutils import paths
import os

imagePaths = list(paths.list_images('classification'))


for imagePath in imagePaths:
    datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range=90)

    imgName = imagePath.split("/")[-1].split(".")[0]
    print(imgName)
    img = cv2.imread(imagePath)

    datagen.fit(img)
    count = 0
    img_batch = datagen.flow(img, batch_size=9)
    for imagem in img_batch:
        cv2.imwrite("newDataset/" + imgName, crop_img)
