import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess_frame(frame):
    frame=cv2.resize(frame,(224,224))
    frame=img_to_array(frame)
    frame= np.expand_dims(frame, axis=0)
    frame=frame/255.0
    return frame 

def load_frames_from_dir(directory):
    frames = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            frame = cv2.imread(os.path.join(directory, filename))
            if frame is not None:
                frame = preprocess_frame(frame)
                frames.append(frame)
    return np.vstack(frames)

#create a folder data and two folders authentic and deepfake in it push the datasets onto it

authentic_frames= load_frames_from_dir('data/authentic/')
deepfake_frames= load_frames_from_dir('data/deepfake/')

authentic_labels = np.zeros(len(authentic_frames))
deepfake_labels = np.ones(len(deepfake_frames))

X= np.concatenate((authentic_frames, deepfake_frames), axis=0)
y= np.concatenate((authentic_labels, deepfake_labels), axis=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

model.save('deepfake_detector_model.h5')
