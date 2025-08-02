import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

# Load dataset (optimized reading)
data = pd.read_csv('fer2013.csv', usecols=['emotion', 'pixels'])

# Faster preprocessing using vectorized operations
pixels = np.array([np.fromstring(pixel, sep=' ', dtype='float32') for pixel in data['pixels']]) / 255.0
pixels = pixels.reshape(-1, 48, 48, 1)

# Prepare labels (unchanged)
emotions = pd.get_dummies(data['emotion']).values

# Split data (smaller test size for more training data)
X_train, X_test, y_train, y_test = train_test_split(
    pixels, emotions, 
    test_size=0.15,  # Reduced from 0.2
    random_state=42,
    stratify=data['emotion']
)

# Build model (added BatchNorm for faster convergence)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),  # Speeds up training
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),  # Additional normalization
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile model (higher initial learning rate)
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Increased from 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Enhanced training with callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
]

# Train model (larger batch size + fewer epochs)
history = model.fit(
    X_train, y_train,
    batch_size=128,  # Increased from 64
    epochs=40,       # Reduced from 50 (early stopping will likely stop earlier)
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Save model (unchanged)
model.save('emotion_model.h5')

# Plot training history (unchanged but now shows LR reduction points)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history.png')
plt.show()