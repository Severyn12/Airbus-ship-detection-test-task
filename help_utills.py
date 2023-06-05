import numpy as np
import PIL
import cv2
import torch
import keras
import os
from torch.utils.data import Dataset
from torchvision import transforms
from tensorflow.keras import layers
import tensorflow as tf

class ImageDataset(Dataset):
    
    def __init__(self, data_folder, img_ids, masks, transform=None):
        """
        A custom PyTorch dataset for image data.
        Args:
            data_folder (str): Path to the folder containing the image data.
            img_ids (numpy array): List of image IDs.
            masks (pandas dataframe): Dictionary or list of masks corresponding to the image IDs.
            transform (callable, optional): Optional data transformation to be applied to the images and masks.
        """
        self.transform = transform
        self.folder_path = data_folder
        self.img_ids = img_ids
        self.masks = masks

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple (tensor_img, label) containing the image tensor and the label tensor.
        """
        path = os.path.join(self.folder_path, self.img_ids[idx])
        try:
            img = PIL.Image.open(path).convert('RGB') 
        except Exception as e:
            raise Exception(f"Failed to load the image with the next ID: {path}") from e
        msk = self.getlabel(self.img_ids[idx])
        tensor_img = transforms.ToTensor()(img)
        tensor_msk = transforms.ToTensor()(msk)
            
        if self.transform is not None:
            tensor_img = self.transform[0](tensor_img).T
            tensor_msk = self.transform[1](tensor_msk).T

        tensor_msk = tensor_msk.to(torch.int)
        label = one_hot_encoder(tensor_msk)
        
        return tensor_img, label
    
    def getlabel(self, img_id):
        """
        Get the label(mask) corresponding to the given image ID.
        Args:
            img_id (str): Image ID for which to retrieve the mask.
        Returns:
            ndarray: Mask as a numpy array.
        """
        encoded_pd = self.masks.loc[self.masks['ImageId'] == img_id, 'EncodedPixels'].values
        if not str(encoded_pd[0]) == 'nan':
            encoded_pd = ' '.join(map(lambda x: x, encoded_pd))
            msk_obj = rle_to_mask(encoded_pd)
        else:
            msk_obj = np.zeros((768,768))
            
        return msk_obj
    
    def getImgandMask(self, idx):
        """
         Get the image, mask, and image ID at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple (img, msk, img_id) containing the image, mask, and image ID.
        """
        img_id = self.img_ids[idx]
        path = os.path.join(self.folder_path, img_id)
        img = np.array(PIL.Image.open(path).convert('RGB'))
        msk = self.getlabel(img_id)
        return img, msk, img_id
    
class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=32, shuffle=True):
        """
        A custom Keras data generator.
        Args:
            dataset (ImageDataset): An instance of the ImageDataset class or any other dataset class.
            batch_size (int, optional): Batch size for generating data.
            shuffle (bool, optional): Whether to shuffle the data at the end of each epoch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Get the total number of batches in the data generator.
        Returns:
            int: Number of batches in the data generator.
        """
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """
        Get a batch of data at the specified index.
        Args:
            index (int): Index of the batch to retrieve.
        Returns:
            tuple: A tuple (inputs, labels) containing the inputs and labels for the batch.
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        inputs = []
        labels = []
        for idx in indexes:
            input_data, target_data = self.dataset[idx]
            inputs.append(input_data/255)  
            labels.append(target_data) 

        inputs = np.stack(inputs)
        labels = np.stack(labels)
        return inputs, labels

    def on_epoch_end(self):
        """
        Shuffle the dataset at the end of each epoch.
        """
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def rle_decoder(encoded_px_str):
    """
    Decode a string of run-length encoded pixels.
    Args:
        encoded_px_str (str): String of encoded pixels in the format "start length start length ...".
    Returns:
        list: Decoded pixels as a list of coordinate tuples (x, y).
    """
    decoded_px = []
    temp_lst = encoded_px_str.split()
    encoded_px = np.array([(int(temp_lst[idx]), int(temp_lst[idx+1])) for idx in range(0, len(temp_lst) - 1, 2)])
    for pair in encoded_px:
        for interm_px in range(pair[0], pair[0]+pair[1]):
            st_position = min(interm_px % 768,766), min(interm_px // 768, 766)
            decoded_px.append(st_position)
            
    return decoded_px

def mask_img(mask_pnts, shape=(768, 768)):
    """
    Create a binary mask image from a list of mask points.
    Args:
        mask_pnts (list): List of coordinate tuples (x, y) representing mask points.
        shape (tuple, optional): Shape of the mask image. Defaults to (768, 768).
    Returns:
        ndarray: Binary mask image as a NumPy array.
    """
    mask_image = np.zeros(shape)
    for point in mask_pnts:
        mask_image[point] = 1
    return mask_image

def one_hot_encoder(mask, num_classes=2):
    """
    One-hot encode a mask image.
    Args:
        mask (ndarray): Mask image as a NumPy array.
        num_classes (int, optional): Number of classes. Defaults to 2.
    Returns:
        ndarray: One-hot encoded mask as a NumPy array.
    """
    return np.squeeze(np.eye(num_classes)[mask])

def rle_to_mask(encoded_pixels):
    """
    Convert run-length encoded pixels to a mask image.
    Args:
        encoded_pixels (str): String of run-length encoded pixels.
    Returns:
        ndarray: Mask image as a NumPy array.
    """
    return mask_img(rle_decoder(encoded_pixels))

def mask_to_rel(mask, shape=(768,768)):
    """
    Convert a binary mask to its corresponding Run-Length Encoding (RLE) representation.
    Args:
        mask (ndarray): Binary mask array.   
    Returns:
        str: RLE representation of the mask.
    """
    if mask.shape != shape:
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[::2] -= 1
    return ' '.join(str(x) for x in runs)

def predict_mask(model, image):
    """
    Generate a prediction mask for a given input image using a trained model.
    Args:
        model (Keras model): Trained model.
        image (ndarray): Input image array.
    Returns:
        ndarray: Predicted mask array.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = np.expand_dims(image, axis=0) / 255.0
    pred_mask = model.predict(image).argmax(axis=-1)[0]
    return pred_mask

def doubConv(x, filters_num):
    """
    Double convolution block.
    Args:
        x (Tensor): Input tensor.
        filters_num (int): Number of filters for the convolution layers.
    Returns:
        Tensor: Output tensor after applying two convolution layers.
    """
    x = tf.keras.layers.Conv2D(filters_num, 3, padding = "same", activation = tf.keras.activations.relu, kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.Conv2D(filters_num, 3, padding = "same", activation = tf.keras.activations.relu, kernel_initializer = "he_normal")(x)
    return x

def downsample(x, filters_num):
    """
    Downsample block.
    Args:
        x (Tensor): Input tensor.
        filters_num (int): Number of filters for the convolution layers.
    Returns:
        Tensor: Output tensor after downsampling.
    """
    conv = doubConv(x, filters_num)
    pool = tf.keras.layers.MaxPool2D(2)(conv)
    pool = tf.keras.layers.Dropout(0.4)(pool)
    return conv, pool

def upsample(x, conv, filters_num):
    """
    Upsample block.
    Args:
        x (Tensor): Input tensor.
        conv (Tensor): Tensor from the corresponding downsample block.
        filters_num (int): Number of filters for the convolution layers.
    Returns:
        Tensor: Output tensor after upsampling.
    """
    x = tf.keras.layers.Conv2DTranspose(filters_num, 3, 2, padding="same")(x)
    x = tf.keras.layers.concatenate([x, conv])
    x = tf.keras.layers.Dropout(0.4)(x)
    x = doubConv(x, filters_num)
    return x

def get_UNET_acrhitecture():
    """
    Creates and returns the UNET's acrhitecture.
    Returns:
      tf.keras.Model: UNET model
    """
    # input layer
    input_layer = tf.keras.layers.Input(shape=(256,256,3))
    # downsampling process
    conv1, pool1 = downsample(input_layer, 64)
    conv2, pool2 = downsample(pool1, 128)
    conv3, pool3 = downsample(pool2, 256)
    conv4, pool4 = downsample(pool3, 512)

    bottleneck = doubConv(pool4, 1024)
    # upsampling process
    uconv4 = upsample(bottleneck, conv4, 512)
    uconv3 = upsample(uconv4, conv3, 256)
    uconv2 = upsample(uconv3, conv2, 128)
    uconv1 = upsample(uconv2, conv1, 64)
    # output layer
    output_layer = tf.keras.layers.Conv2D(2, 1, padding="same", activation = tf.keras.activations.softmax)(uconv1)
    UNET = tf.keras.Model(input_layer, output_layer)

    return UNET

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Compute the Dice coefficient between ground truth and predicted masks.
    Args:
        y_true : Ground truth mask.
        y_pred : Predicted mask.
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        Dice coefficient.
    """
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice