
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0 
    return frame

def load_frames_from_dir(directory):
    frames = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            frame = cv2.imread(os.path.join(directory, filename))
            if frame is not None:
                frame = preprocess_frame(frame)
                frames.append(frame)
    return np.array(frames)

model = load_model('deepfake_detector_model.h5')

test_frames = load_frames_from_dir('data/test/')

test_labels = np.array([0] * len(test_frames)) 

loss, accuracy = model.evaluate(test_frames, test_labels)
print(f'Test accuracy: {accuracy:.4f}')
