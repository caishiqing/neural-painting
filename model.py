from tensorflow.keras import layers
import tensorflow as tf
import types


def RenderNet(param_size, canvas_width):
    fmap_size = canvas_width // 8
    model = tf.keras.Sequential(
        layers=[
            layers.Input((param_size,), dtype=tf.float32),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(fmap_size * fmap_size * 12),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Reshape([canvas_width//4, canvas_width//4, 3]),
            layers.Conv2DTranspose(8, 3, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(3, 3, 2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('sigmoid')
        ]
    )
    model.save = types.MethodType(save_model, model)
    return model


def save_model(cls, filepath, **kwargs):
    kwargs["include_optimizer"] = False
    tf.keras.models.save_model(cls, filepath, **kwargs)