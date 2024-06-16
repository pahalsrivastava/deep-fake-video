
# Deepfake Video Detector

This project demonstrates how to build and train a deepfake video detector using a pre-trained VGG16 model in TensorFlow/Keras.

## Overview

Deepfake technology has become increasingly sophisticated, making it crucial to develop tools to detect manipulated videos. This project uses machine learning techniques to distinguish between authentic and deepfake videos based on frames extracted from the videos.

## Features

- Trains a deepfake detector using a pre-trained VGG16 convolutional neural network.
- Evaluates the trained model's accuracy on a test set of video frames.
- Provides scripts for training (`train.py`) and evaluation (`model.py`).

## Dependencies

- Python
- TensorFlow 
- NumPy
- OpenCV (cv2)
- Scikit-learn

Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
