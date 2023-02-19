from tensorflow.keras import layers
import tensorflow as tf
import types


# def RenderNet(param_size, canvas_width):
#     fmap_size = canvas_width // 8
#     model = tf.keras.Sequential(
#         layers=[
#             layers.Input((param_size,), dtype=tf.float32),
#             layers.Dense(512, activation='relu'),
#             layers.Dense(1024, activation='relu'),
#             layers.Dense(2048, activation='relu'),
#             layers.Dense(fmap_size * fmap_size * 12),
#             layers.BatchNormalization(),
#             layers.Activation('relu'),
#             layers.Reshape([canvas_width//4, canvas_width//4, 3]),
#             layers.Conv2DTranspose(8, 3, 2, padding='same', activation='relu'),
#             layers.Conv2DTranspose(3, 3, 2, padding='same'),
#             layers.BatchNormalization(),
#             layers.Activation('sigmoid')
#         ]
#     )
#     model.save = types.MethodType(save_model, model)
#     return model


class PixelShuffle(layers.Layer):
    def __init__(self, upscale_factor, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config['upscale_factor'] = self.upscale_factor
        return config

    def build(self, input_shape):
        b, h, w, c = input_shape
        assert (c // self.upscale_factor) % self.upscale_factor == 0
        self.output_channels = c // self.upscale_factor // self.upscale_factor
        super(PixelShuffle, self).build(input_shape)

    def _phase_shift(self, x):
        b, h, w, _ = x.get_shape().as_list()
        r = self.upscale_factor
        x = tf.reshape(x, (-1, h, w, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, h, 1)  # h, [b, 1, w, r, r]
        x = tf.concat(x, axis=3)  # b, 1, w, h*r, r
        x = tf.squeeze(x, 1)  # b, w, h*r, r
        x = tf.split(x, w, 1)  # w, [b, 1, h*r, r]
        x = tf.concat(x, axis=3)  # b, 1, h*r, w*r
        x = tf.squeeze(x, 1)  # b, h*r, w*r
        return tf.expand_dims(x, -1)

    def call(self, inputs):
        xs = tf.split(inputs, self.output_channels, -1)
        xs = [self._phase_shift(x) for x in xs]
        return tf.concat(xs, axis=-1)

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        r = self.upscale_factor
        return b, h*r, w*r, c//(r**2)


def RenderNet(param_size, canvas_width):
    fmap_size = canvas_width // 8

    x = layers.Input((param_size,), dtype=tf.float32)
    h = layers.Dense(512, activation='relu')(x)
    h = layers.Dense(1024, activation='relu')(h)
    h = layers.Dense(2048, activation='relu')(h)
    h = layers.Dense(fmap_size * fmap_size * 16)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = layers.Reshape([fmap_size, fmap_size, 16])(h)
    h = layers.Conv2D(32, 3, padding='same', activation='relu')(h)
    h = layers.Conv2D(32, 3, padding='same')(h)
    h = PixelShuffle(2)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(16, 3, padding='same', activation='relu')(h)
    h = layers.Conv2D(16, 3, padding='same')(h)
    h = PixelShuffle(2)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(8, 3, padding='same', activation='relu')(h)
    h = layers.Conv2D(4*6, 3, padding='same')(h)
    h = PixelShuffle(2)(h)
    foreground, stroke_alpha_map = layers.Lambda(lambda x: tf.split(x, 2, axis=3))(h)

    model = tf.keras.Model(inputs=x, outputs=[foreground, stroke_alpha_map])
    model.save = types.MethodType(save_model, model)
    return model


def save_model(cls, filepath, **kwargs):
    kwargs["include_optimizer"] = False
    tf.keras.models.save_model(cls, filepath, **kwargs)


tf.keras.utils.get_custom_objects().update(
    {'PixelShuffle': PixelShuffle}
)
