"""
LETTER RECOGNITION WITH NEURAL NETWORKS
========================================

HOW IT WORKS:
1. LOAD DATA: Read images of handwritten letters (A-Z) and their labels
2. PREPROCESS: Normalize pixel values, split into train/test sets
3. BUILD MODEL: Create a Convolutional Neural Network (CNN)
4. TRAIN: Show the model thousands of examples so it learns patterns
5. EVALUATE: Test on unseen data to measure real-world performance

NEURAL NETWORK BASICS:
- A neural network is a function that takes an input (image) and outputs a prediction (letter)
- It consists of layers of "neurons" that each do: output = activation(weights * input + bias)
- During training, we adjust weights to minimize prediction errors (loss)
- The "learning" happens through backpropagation: calculate gradients and update weights
"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
print("Modules loaded")


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
# Load images from folders (A/, B/, C/, ..., Z/)
# Each image is a 32x32 pixel grayscale image of a handwritten letter
# Labels are integers 0-25 representing A-Z

from letters.get_data import load_tensorflow_data
features, labels = load_tensorflow_data()
print("Data loaded")
print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

# =============================================================================
# STEP 2: PREPROCESS DATA
# =============================================================================

# Split data into training set (80%) and test set (20%)
# WHY? We train on one set and test on another to measure generalization
# If we tested on training data, we'd just measure memorization, not learning
# random_state=42 ensures reproducible splits
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Debug: Check label distribution (should be roughly equal per letter)
unique, counts = np.unique(y_train, return_counts=True)
print(f"Label distribution: {dict(zip(unique, counts))}")
print(f"Labels range: {y_train.min()} to {y_train.max()}")

# Debug: Visualize a few samples to verify data looks correct
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]} ({chr(65 + y_train[i])})")
    ax.axis('off')
plt.suptitle("Sample training images (after inversion)")
plt.tight_layout()
plt.savefig("debug_samples.png")
plt.show()

# Reshape for CNN: add channel dimension
# Before: (N, 32, 32) - just height and width
# After: (N, 32, 32, 1) - height, width, and 1 color channel (grayscale)
# WHY? Conv2D expects 4D input: (batch_size, height, width, channels)
X_train = X_train.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)


# =============================================================================
# STEP 3: BUILD THE MODEL (CNN Architecture)
# =============================================================================
"""
CONVOLUTIONAL NEURAL NETWORK (CNN) EXPLAINED:

Unlike regular Dense networks that treat each pixel independently,
CNNs understand spatial relationships - they know pixels next to each other
form edges, curves, and shapes.

LAYER TYPES:
- Conv2D: Slides small filters (3x3) across the image to detect features
  - Early layers detect simple features: edges, corners
  - Later layers detect complex features: curves, letter parts
  
- MaxPooling2D: Shrinks the image by keeping max value in each 2x2 region
  - Reduces computation
  - Provides translation invariance (letter can shift slightly)
  
- Flatten: Converts 2D feature maps to 1D vector for classification

- Dense: Traditional fully-connected layer for final classification
  - Each neuron connects to all neurons in previous layer
  
- Dropout: Randomly disables neurons during training
  - Prevents overfitting (memorizing training data)
  - Forces network to learn robust features

- Softmax: Converts raw scores to probabilities (0-1, sum to 1)
"""

model = Sequential([
    # Input layer: expects 32x32 grayscale images
    Input(shape=(32, 32, 1)),
    
    # CONV BLOCK 1
    # Conv2D(16 filters, 3x3 kernel): Learns 16 different 3x3 patterns
    # Each filter might detect: horizontal edge, vertical edge, corner, etc.
    # padding='same' keeps output same size as input
    # activation='relu' = max(0, x) - introduces non-linearity, enables complex patterns
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    # Output: 32x32x16 (16 feature maps)
    
    # MaxPooling: Take max value in each 2x2 region
    # Reduces 32x32 → 16x16 (halves spatial dimensions)
    MaxPooling2D((2, 2)),
    # Output: 16x16x16
    
    # CONV BLOCK 2
    # More filters (32) to detect more complex patterns
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Output: 16x16x32
    
    MaxPooling2D((2, 2)),
    # Output: 8x8x32
    
    # Flatten: Convert 3D feature maps to 1D vector
    # 8 * 8 * 32 = 2048 values
    Flatten(),
    # Output: 2048
    
    # Dense layer: Learn combinations of features
    # 64 neurons, each looks at all 2048 inputs
    Dense(64, activation='relu'),
    # Output: 64
    
    # Dropout: During training, randomly set 30% of neurons to 0
    # This prevents overfitting by forcing redundant learning
    Dropout(0.3),
    
    # Output layer: 26 neurons (one per letter A-Z)
    # Softmax activation converts outputs to probabilities
    # Example output: [0.01, 0.02, 0.90, 0.01, ...] = "C" with 90% confidence
    Dense(26, activation='softmax')
])


# =============================================================================
# STEP 4: COMPILE THE MODEL
# =============================================================================
"""
COMPILATION sets up HOW the model learns:

OPTIMIZER (Adam):
- Algorithm that updates weights based on gradients
- Adam = "Adaptive Moment Estimation" - adjusts learning rate per parameter
- learning_rate=0.001 = step size for weight updates (smaller = slower but stable)

LOSS FUNCTION (sparse_categorical_crossentropy):
- Measures how wrong the predictions are
- "Sparse" = labels are integers (0, 1, 2...) not one-hot vectors
- "Categorical" = multi-class classification
- "Crossentropy" = -log(probability of correct class)
- Lower loss = better predictions

METRICS (accuracy):
- What we monitor during training
- accuracy = (correct predictions) / (total predictions)
"""

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model architecture summary
model.summary()


# =============================================================================
# STEP 5: SET UP TRAINING CALLBACKS
# =============================================================================
"""
CALLBACKS are functions called during training:

EarlyStopping:
- Monitors validation loss (error on data the model hasn't seen)
- If val_loss doesn't improve for 10 epochs, stop training
- restore_best_weights=True: go back to the best model found
- WHY? Prevents overfitting - training too long memorizes training data

ReduceLROnPlateau:
- If val_loss plateaus for 5 epochs, reduce learning rate by 50%
- Smaller learning rate = finer adjustments = can escape local minima
- min_lr prevents learning rate from becoming too small
"""

early_stop = EarlyStopping(
    monitor='val_loss',      # Watch validation loss
    patience=10,             # Wait 10 epochs before stopping
    restore_best_weights=True,  # Use best weights, not last
    verbose=1                # Print when triggered
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # Watch validation loss
    factor=0.5,              # Multiply LR by 0.5 when triggered
    patience=5,              # Wait 5 epochs before reducing
    min_lr=1e-6,             # Don't go below 0.000001
    verbose=1                # Print when triggered
)


# =============================================================================
# STEP 6: TRAIN THE MODEL
# =============================================================================
"""
TRAINING PROCESS:

For each epoch (full pass through training data):
  For each batch of 16 images:
    1. FORWARD PASS: Feed images through network, get predictions
    2. CALCULATE LOSS: Compare predictions to true labels
    3. BACKWARD PASS (Backpropagation): Calculate gradients
       - How much each weight contributed to the error
    4. UPDATE WEIGHTS: Adjust weights to reduce error
       - new_weight = old_weight - learning_rate * gradient

validation_split=0.15: Use 15% of training data for validation
- Model never trains on validation data
- Helps detect overfitting (train acc high, val acc low)
"""

history = model.fit(
    X_train, y_train,        # Training data
    epochs=100,              # Maximum passes through the data
    batch_size=16,           # Process 16 images at a time
    validation_split=0.15,   # Use 15% for validation
    callbacks=[early_stop, reduce_lr],  # Stop early if needed
    verbose=1                # Show progress
)


# =============================================================================
# STEP 7: EVALUATE THE MODEL
# =============================================================================
"""
EVALUATION on test set:

The test set was NEVER seen during training.
Test accuracy tells us how well the model generalizes to new data.

Common patterns:
- Train acc high, test acc high → Good model!
- Train acc high, test acc low → Overfitting (memorized training data)
- Train acc low, test acc low → Underfitting (model too simple or needs more data)
"""

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history_cnn.png")
plt.show()

"""
# Get predictions for confusion matrix
y_pred = model.predict(X_test)  # Raw probabilities for each class
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert to class labels (0-25)

# Confusion matrix shows which letters get confused with which
# Rows = true labels, Columns = predicted labels
# Diagonal = correct predictions
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)
"""

# Save model for later use
model.save("./models/CNN_model.keras")
print("Model saved to 'CNN_model.keras'")
