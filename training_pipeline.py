import numpy as np
import pandas as pd
from help_utills import ImageDataset, DataGenerator, get_UNET_acrhitecture, dice_coefficient
from torchvision import transforms
from sklearn.model_selection import train_test_split
from keras.metrics import Recall
import tensorflow as tf




TRAIN_DATA_PATH = '/kaggle/input/airbus-ship-detection/train_v2'
TEST_DATA_PATH = '/kaggle/input/airbus-ship-detection/test_v2'
IMAGES_WITHOUT_SHIPS = 20000
IMAGES_WITH_SHIPS = 25000

output_size = (256, 256)
rescale_transform = transforms.Resize(output_size)
composed_transform = rescale_transform,rescale_transform

MASKS_DF = pd.read_csv("/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv")
MASKS_DF['Ship_class'] = MASKS_DF['EncodedPixels'].notnull().astype(int)
#Deleting the corrupted image
MASKS_DF = MASKS_DF.drop(MASKS_DF[MASKS_DF['ImageId'] == '6384c3e78.jpg'].index)

no_ship_imgs = MASKS_DF[MASKS_DF['Ship_class'] == 0].sample(frac=1).sample(n=IMAGES_WITHOUT_SHIPS)
no_ship_imgs = no_ship_imgs.drop('Ship_class', axis=1)
no_ship_imgs.reset_index(drop=True,inplace=True)

ship_imgs = MASKS_DF[MASKS_DF['Ship_class'] == 1]
ship_imgs = ship_imgs.groupby('ImageId')[['EncodedPixels']].agg(lambda rle_encoded_px: ' '.join(rle_encoded_px)).sample(frac=1).sample(n=IMAGES_WITH_SHIPS)
ship_imgs.reset_index(inplace=True)


all_images = np.concatenate((ship_imgs['ImageId'].values, no_ship_imgs['ImageId'].values))
train, test_ids = train_test_split(all_images, test_size=0.1, random_state=42)
train_ids, dev_ids = train_test_split(train, test_size=0.11, random_state=42)

train_data = ImageDataset(TRAIN_DATA_PATH, train_ids, MASKS_DF, transform=composed_transform)
dev_data = ImageDataset(TRAIN_DATA_PATH, dev_ids, MASKS_DF, transform=composed_transform)
test_data = ImageDataset(TRAIN_DATA_PATH, test_ids, MASKS_DF, transform=composed_transform)

train_generator = DataGenerator(train_data)  
dev_generator = DataGenerator(dev_data)
test_generator = DataGenerator(test_data)
recall = Recall()

model = get_UNET_acrhitecture()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', dice_coefficient, recall])
tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient

model.fit(train_generator, validation_data=dev_generator, epochs=2)
model.save('model-UNET.h5')