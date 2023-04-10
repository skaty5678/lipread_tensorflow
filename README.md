# Lip Reading Using Deep Learning

This project aims to build a deep learning model that can recognize spoken words from a person's lip movements. The model takes in a video of a person speaking and outputs the transcribed text.



## Requirements

The following libraries are required to run the code:

* opencv-python
* matplotlib
* imageio
* gdown
* tensorflow


You can install the required libraries using pip by running the following command:

### *!pip install opencv-python matplotlib imageio gdown tensorflow*



## Dataset
The dataset used for training and evaluation is the GRID corpus, which is a collection of audiovisual recordings of people speaking short sentences.



## Preprocessing

The preprocessing steps include:

1. Extracting the video frames and preprocessing them by converting them to grayscale and cropping them to focus only on the lips.
2. Loading the text alignments for the videos and converting them into numerical values.



## Model Architecture

The model architecture consists of the following layers:

* Conv3D: extracts spatiotemporal features from the video frames
* BatchNormalization: normalizes the output of the Conv3D layer
* SpatialDropout3D: drops out entire feature maps from the Conv3D output to reduce overfitting
* Reshape: flattens the output of the Conv3D layer
* Bidirectional LSTM: processes the flattened Conv3D output to capture the temporal context of the video frames
* Dense: produces the final output by classifying the spoken word
* Activation: applies the softmax function to the output of the Dense layer



## Training

The model is trained using the Adam optimizer and the categorical cross-entropy loss function. The model is trained on 450 videos and validated on the remaining 50 videos.



## Conclusion

In conclusion, this project demonstrates the feasibility of using deep learning to recognize spoken words from a person's lip movements. The lip reading performance can be improved by using a larger dataset.