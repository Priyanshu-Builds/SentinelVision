import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from model.optimized_cnn import optimized_model

def generate_dummy_data(num_samples=1000, img_shape=(64, 64, 3)):
    # Generate random image data and binary labels
    X = np.random.rand(num_samples, *img_shape).astype("float32")
    y = np.random.randint(0, 2, size=(num_samples, 1))
    return X, y

def train_model():
    X, y = generate_dummy_data(num_samples=1000)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    model = optimized_model(input_shape=(64, 64, 3))
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_val, y_val), callbacks=callbacks)
    
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f} | Validation Accuracy: {accuracy:.4f}")
    
    model.save("optimized_model.h5")
    
if __name__ == "__main__":
    train_model()
