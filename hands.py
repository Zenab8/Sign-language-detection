import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tqdm import tqdm
import re

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def preprocess_hand_image(image_path, output_path):
    image = cv2.imread(image_path)
    
    h, w = image.shape[:2]
    if h < 200 or w < 200:
        scale_factor = max(200 / h, 200 / w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            h, w, _ = image.shape
            landmark_points = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark])
            x_min, y_min = np.min(landmark_points, axis=0).astype(int)
            x_max, y_max = np.max(landmark_points, axis=0).astype(int)
            
            padding = 20
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
            
            cropped = image[y_min:y_max, x_min:x_max]
            cropped_resized = cv2.resize(cropped, (224, 224))
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cropped_resized)


def simple_preprocess(image_path, output_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (224, 224))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized)


def sanitize_label(label):
    if isinstance(label, tf.Tensor):
        label = label.numpy()
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    elif isinstance(label, np.ndarray):
        if label.ndim > 0:
            label = str(np.argmax(label)) if len(label) > 1 else str(int(label[0]))
        else:
            label = str(int(label))
    elif np.isscalar(label):
        label = str(int(label))
    else:
        label = str(label)
    
    label = re.sub(r'[<>:"/\\|?*\[\]\s]+', "_", label)
    return label.strip("_") or "unknown"


def preprocess_dataset_tf(dataset, output_dir="train_ready", use_hand_detection=False):
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    
    print(f"Starting preprocessing to: {output_dir}")
    print(f"Hand detection: {'ENABLED' if use_hand_detection else 'DISABLED'}")
    
    dataset_unbatched = dataset.unbatch()
    
    for image, label in tqdm(dataset_unbatched, desc="Processing images"):
        temp_input_path = f"temp_input_{processed}.jpg"
        
        image_np = image.numpy()
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        
        cv2.imwrite(temp_input_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        
        label_value = sanitize_label(label)
        label_dir = os.path.join(output_dir, label_value)
        os.makedirs(label_dir, exist_ok=True)
        
        output_path = os.path.join(label_dir, f"image_{processed:06d}.jpg")
        
        if use_hand_detection:
            preprocess_hand_image(temp_input_path, output_path)
        else:
            simple_preprocess(temp_input_path, output_path)
        
        os.remove(temp_input_path)
        processed += 1
    
    print(f"\nSuccessfully processed: {processed} images")


def visualize_samples(output_dir, n_samples=5):
    all_images = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                all_images.append(os.path.join(root, f))
    
    sample_paths = random.sample(all_images, min(n_samples, len(all_images)))
    
    plt.figure(figsize=(15, 5))
    for i, path in enumerate(sample_paths):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(sample_paths), i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Label: {os.path.basename(os.path.dirname(path))}")
    
    plt.tight_layout()
    plt.show()


def get_dataset_stats(output_dir):
    stats = {}
    total_images = 0
    
    for label in os.listdir(output_dir):
        label_path = os.path.join(output_dir, label)
        if os.path.isdir(label_path):
            image_count = len([f for f in os.listdir(label_path) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            stats[label] = image_count
            total_images += image_count
    
    print(f"\nDataset Statistics for: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"Number of classes: {len(stats)}")
    print(f"\nImages per class:")
    for label, count in sorted(stats.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        print(f"  {label}: {count} images")


def preprocess_from_folder(input_dir, output_dir, use_hand_detection=True):
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    
    all_images = []
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append((os.path.join(class_path, img_file), class_folder))
    
    print(f"Found {len(all_images)} images to process")
    
    for img_path, label in tqdm(all_images, desc="Processing images"):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        output_path = os.path.join(label_dir, os.path.basename(img_path))
        
        if use_hand_detection:
            preprocess_hand_image(img_path, output_path)
        else:
            simple_preprocess(img_path, output_path)
        
        processed += 1
    
    print(f"\nSuccessfully processed: {processed} images")


if __name__ == "__main__":
    preprocess_from_folder(
        input_dir="asl_alphabet/asl_alphabet_train/asl_alphabet_train",
        output_dir="train_ready_skeleton",
        use_hand_detection=True
    )
    
    get_dataset_stats("train_ready_skeleton")
    visualize_samples("train_ready_skeleton", n_samples=5)