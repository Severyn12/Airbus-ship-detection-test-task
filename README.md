# Airbus-ship-detection-test-task
***
## Aim
The main goal of this task was to develop an efficient algorithm, that will automatically identify and detect ships in satellite images. In order to achieve this, was decided to use a semantic segmentation model, named UNET. 
***
## Dataset observation
The dataset is in public access: https://www.kaggle.com/competitions/airbus-ship-detection/data

Dtatset itself consisted of several main files. Firstly, there were two folders: "train" containing images used for model development, and "test" for evaluation and result submission. Additionally, there was a file with corresponding images' masks, which were encoded in the RLE(Run-Length Encoding) format.  Speaking about RLE encoding, it's important to mention that it represents consecutive occurrences of pixels with the same value (e.g., ship or background) as pairs of values: the starting position and the length of the run. For example, "8 5" in my case indicated a sequence of 1s(ship class) starting at position 8 with a length of 5. 

For further work, those string masks' representations were decoded into pixel-level ones. This process involved converting the RLE strings into arrays, where the positions specified by the encoded values were marked as the corresponding mask value. After that, those mask arrays were one-hot encoded and used as labels for our samples.

### Data preprocessing
Besides the decoding and one-hot encoding the masks, was also made some preprocessing steps for input image data. To be more specific all images were reshaped into size 256x256x3 pixels and then scaled to fall within the range of [0, 1]. The important remark here would be that reshaped operation was also applied for decoded masks' objects before one-hot encoding.

### EDA summary
<img src="https://github.com/Severyn12/qwert/assets/73779019/aaa5b043-cea7-42a0-9789-f992ff448211" alt="EDA summary" width="700">

The key observation, that was made during EDA, is that our dataset is unbalanced, there prevail images without ships. That is a problem, which becomes even worse if think in terms of pixels because even on those images that contain ships the percentage of pixels occupied by the ships is relatively small compared to the overall image. To address this issue, while creating a model's dataset, I ensured that it contained an approximately equal number of images from each class, with a slightly higher proportion of images containing ships.

**Remark:** More detailed EDA with corresponding plots and code can be observed in the file **EDA_work**

## Architecture
In order to accomplish this work, I used a UNET architecture, which shows great performance for various image segmentation tasks by effectively capturing both local and global image information. 

The UNET architecture mainly consists of two parts: encoder and decoder components. The encoder gradually reduces the spatial dimensions of the input image while increasing the number of feature channels. This process captures high-level semantic information, while the decoder works with the encoded features and gradually upsamples them and at the same time reduces the number of channels. This helps recover the spatial resolution and generate segmentation masks that align with the original input image. The key feature of the UNet model is the skip connections, which allows us to fight the gradient vanishing problem. 

**Below you could observe the UNET's architecture:**
<img src="https://github.com/Severyn12/qwert/assets/73779019/98969cb4-23b2-4c80-a090-939e52c1cbaf" alt="UNET" width="750">

## Approach
The UNET was trained using a supervised approach, where we provide the model with labeled input data and then it learns from it some patterns. Also, I split my dataset into three parts: training, development(validation), and test sets. On the first two, I actually was training and validating my model architecture, and based on the last one I was evaluating my model's overall performance, by calculating the accuracy, dice score, and recall metrics, which are good choice metrics for segmentation tasks. For example, the recall was chosen to measure the ability of the model to correctly identify all positive instances in the image, which is important for us because of class imbalance(we have much more instances of the zero class compared to the first class).

Bellow you could observe plots, that illustrate the model's performance:
<img src="https://github.com/Severyn12/qwert/assets/73779019/7ecd53aa-808d-427e-a5be-89fb45339972" alt="Performance plots" width="700">
<img src="https://github.com/Severyn12/qwert/assets/73779019/d4a43348-94c8-4983-9dc8-95ce64d51523" alt="Mask example" width="700">

## File navigation:
* **Airbus-ship-detection_EDA.ipynb** - this file contains the exploratory data analysis.
* **training_pipeline.py** - contains the model's training pipeline. 
* **inference_pipeline.py** - contains the model's evaluation on the test set pipeline.
* **Airbus-ship-detection.ipynb** - the notebook file with model training and evaluation.

**NOTE**: before running the **training_pipeline.py** or **inference_pipeline.py** you should specify the correct path to the data. In the case of model **inference_pipeline.py** you should also specify the correct path to the saved model.
