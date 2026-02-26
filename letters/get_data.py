import numpy as np
import cv2
import os

# Define max number of features per folder
max_features = 8000

def load_data():
    features = []
    labels = []
    
    # List all valid single-character folders inside "Letters"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of get_data.py
    print(base_dir)
    folder_list = sorted([f for f in os.listdir(base_dir) if len(f) == 1 and os.path.isdir(os.path.join(base_dir, f))])
    
    if not folder_list:
        print("Error: No valid letter folders found in 'Letters' directory.")
        return None, None

    for index, folder in enumerate(folder_list):
        folder_path = os.path.join(base_dir, folder)
        label = index  # Store index as label

        # Get file list
        file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        if not file_list:
            print(f"Warning: No files found in '{folder_path}', skipping folder.")
            continue  

        # Lists to store images and labels for current folder
        image_list = []
        label_list = []

        for filename in file_list[:max_features]:  # Limit number of files processed
            image_path = os.path.join(folder_path, filename)

            # Read image in grayscale mode
            image = cv2.imread(image_path, 0)  # 0 = grayscale

            if image is None:
                print(f"Warning: Skipping '{filename}', failed to load.")
                continue  

            # Resize the image to 32x32
            resized_img = cv2.resize(image, (32, 32))

            # Append to lists
            image_list.append(resized_img)
            label_list.append([label])  

        # Skip folders where no valid images were loaded
        if not image_list:
            print(f"Warning: No valid images loaded for folder '{folder}', skipping.")
            continue

        # Convert lists to NumPy arrays
        image_dataset = np.array(image_list, dtype=np.float32)  # Shape: (N, 32, 32)
        label_dataset = np.array(label_list, dtype=str)  # Shape: (N, 1)

        # Append datasets
        features.append(image_dataset)
        labels.append(label_dataset)

    if not features:
        print("Error: No valid data loaded.")
        return None, None

    # Concatenate all data into single NumPy arrays
    features = np.concatenate(features, axis=0)  # Shape: (Total_N, 32, 32)
    labels = np.concatenate(labels, axis=0)  # Shape: (Total_N, 1)
    print(f"Loaded {features.shape}")
    return features, labels

def load_tensorflow_data():
    """
    Load and prepare image data for TensorFlow processing.
    
    Returns:
        features: numpy array of shape (N, 32, 32) with normalized pixel values [0, 1]
        labels: numpy array of shape (N, 1) with integer class labels
    """
    # Load raw data from get_data module
    print("Loading data from letters directory...")
    features, labels = load_data()
    
    # Check if data was loaded successfully
    if features is None or labels is None:
        print("Error: Failed to load data.")
        return None, None
    
    print(f"Loaded {features.shape[0]} images with shape {features.shape[1]}x{features.shape[2]}")
    
    # Normalize pixel values from [0, 255] to [0, 1]
    # This is required for TensorFlow neural networks to work effectively
    features_normalized = features / 255.0
    
    print(f"Normalized pixel values to range [{features_normalized.min():.2f}, {features_normalized.max():.2f}]")
    
    # Invert images (black letters on white → white letters on black)
    # This is important because neural networks work better with white-on-black
    features_normalized = 1.0 - features_normalized

    # Convert labels from string to integer type
    # TensorFlow requires numeric labels for classification
    labels_int = labels.astype(np.int32).flatten()  # Shape: (N,)
    
    print("\nData preparation complete!")
    print(f"  - Images: {features_normalized.shape[0]} samples, {features_normalized.shape[1]}x{features_normalized.shape[2]} pixels")

    print(f"Features shape: {features_normalized.shape} (N x R x C)")
    print(f"Labels shape: {labels_int.shape} (N x 1)")
    print(f"Number of classes: {len(np.unique(labels_int))}")
    return features_normalized, labels_int
