"""
Sequential Feedforward Neural Network for Letter Recognition.

This script creates a simple feedforward network (no convolutional layers)
to classify handwritten letters A-Z.
"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("Modules loaded")

# Load prepared data from get_data module
from letters.get_data import load_tensorflow_data
features, labels = load_tensorflow_data()
print("Data loaded")
print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, shuffle=True
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Check label distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"Label distribution: {dict(zip(unique, counts))}")
print(f"Labels range: {y_train.min()} to {y_train.max()}")

# Build Sequential Feedforward Neural Network
# Input: 32x32 = 1024 pixels (flattened)
# Hidden layers: Dense layers with ReLU activation
# Output: 26 classes (A-Z) with softmax activation
model = Sequential([
    Input(shape=(32, 32)),
    
    # Flatten the 2D image to 1D vector (32*32 = 1024 features)
    Flatten(),
    
    # First hidden layer
    Dense(256, activation='relu'),
    Dropout(0.3),
    
    # Second hidden layer
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    # Third hidden layer
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    # Output layer: 26 classes (A-Z)
    Dense(26, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Callbacks for better training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# Train the model
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

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
plt.savefig("training_history_feedforward.png")
plt.show()

# Save the trained model
model.save("./models/feedforward_model_bigdata.keras")
print("Model saved to 'feedforward_model_bigdata.keras'")
