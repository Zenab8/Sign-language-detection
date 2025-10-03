import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_gesture_model(model_path: str):
    # If model_path points to a folder (SavedModel), load_model works too.
    return load_model(model_path)

def preprocess_frame(frame, img_size=(64,64)):
    # Get height and width
    h, w, _ = frame.shape
    
    # Crop the center square (hand is usually there in webcam demos)
    min_dim = min(h, w)
    start_x = w//2 - min_dim//2
    start_y = h//2 - min_dim//2
    cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # Resize and normalize
    img = cv2.resize(cropped, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

