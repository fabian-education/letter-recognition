from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
print("Modules loaded")


from letters.get_data import load_tensorflow_data
features, labels = load_tensorflow_data()
print("Data loaded")
print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Debug: Check label distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"Label distribution: {dict(zip(unique, counts))}")
print(f"Labels range: {y_train.min()} to {y_train.max()}")

# Debug: Visualize a few samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]} ({chr(65 + y_train[i])})")
    ax.axis('off')
plt.suptitle("Sample training images (after inversion)")
plt.tight_layout()
plt.savefig("debug_samples.png")
plt.show()

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)

# Simple model for small dataset (520 samples is VERY small)
# Complex CNN overfits immediately - use simpler architecture
model = Sequential([
    Input(shape=(32, 32, 1)),
    
    # Single Conv Block
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Dense layers (small)
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')     # out = 26 classes (A-Z)
])

# Compile with optimal settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks for better training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,  # Smaller batch for small dataset
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

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
# Make predictions and show confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)
"""

# Save model for later use
model.save("./models/CNN_model.keras")
print("Model saved to 'CNN_model.keras'")
