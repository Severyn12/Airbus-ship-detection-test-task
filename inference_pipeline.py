import tensorflow as tf
import pandas as pd
import PIL
import os
from help_utills import predict_mask, mask_to_rel

TEST_DATA_PATH = '/kaggle/input/airbus-ship-detection/test_v2'
model = tf.keras.models.load_model('/kaggle/working/model-UNET.h5')
predictions_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

for filename in os.listdir(TEST_DATA_PATH):
    file_path = os.path.join(TEST_DATA_PATH, filename)
    image = PIL.Image.open(file_path)
    image = image.resize((256, 256))  

    pred_mask = predict_mask(model, image)
    rle_mask = mask_to_rel(pred_mask)

    predictions_df = predictions_df.append({'ImageId': filename, 'EncodedPixels': rle_mask}, ignore_index=True)

predictions_df.to_csv('sample_submission.csv', index=False)