import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, DepthwiseConv2D, Conv2D, BatchNormalization,
                                     ReLU, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D)
from tensorflow.keras.regularizers import l2

def optimized_model(input_shape=(64, 64, 3), dropout_rate=0.5, weight_decay=1e-4):
    inputs = Input(shape=input_shape)
    x = inputs

    # Block 1
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1,
                        kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = Conv2D(32, kernel_size=(1, 1), padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1,
                        kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = Conv2D(64, kernel_size=(1, 1), padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1,
                        kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = Conv2D(128, kernel_size=(1, 1), padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    # Global Average Pooling and Dropout
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)

    # Output layer: Binary classification using sigmoid
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = optimized_model()
    model.summary()
