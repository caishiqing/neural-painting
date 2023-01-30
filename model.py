import tensorflow as tf
from tensorflow.keras import layers


def RenderNet(param_size, canvas_width):
    fmap_size = canvas_width // 8
    return tf.keras.Sequential(
        layers=[
            layers.Input((param_size,), dtype=tf.float32),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(fmap_size * fmap_size * 24),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Reshape([param_size//4, param_size//4, 3]),
            layers.Conv2DTranspose(12, 3, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(6, 3, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(3, 3, 2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('sigmoid')
        ]
    )
