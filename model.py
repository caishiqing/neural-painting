from tensorflow.keras import layers
from loss import PixelLoss
import tensorflow as tf


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


def rasterNet(x):
    h = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    h = layers.Conv2D(64, 3, padding='same')(h)
    h = PixelShuffle(2)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = layers.Conv2D(32, 3, padding='same', activation='relu')(h)
    h = layers.Conv2D(32, 3, padding='same')(h)
    h = PixelShuffle(2)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = layers.Conv2D(16, 3, padding='same', activation='relu')(h)
    h = layers.Conv2D(16, 3, padding='same')(h)
    h = PixelShuffle(2)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)

    y = layers.Dense(1, activation='sigmoid', name='Raster')(h)
    return y


def shadeNet(x):
    # h = layers.Conv2DTranspose(128, 3, 2, padding='same')(x)
    # h = layers.BatchNormalization()(h)
    # h = layers.Activation('relu')(h)
    # h = layers.Conv2DTranspose(64, 3, 2, padding='same')(h)
    # h = layers.BatchNormalization()(h)
    # h = layers.Activation('relu')(h)
    h = layers.Conv2DTranspose(32, 3, 2, padding='same')(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = layers.Conv2DTranspose(16, 3, 2, padding='same')(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = layers.Conv2DTranspose(8, 3, 2, padding='same')(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)

    y = layers.Dense(3, name='Shade')(h)
    return y


def renderNet(param_size, canvas_width=128):
    x = layers.Input((param_size,), dtype=tf.float32)
    h = layers.Dense(512, activation='relu')(x)
    h = layers.Dense(1024, activation='relu')(h)
    h = layers.Dense(2048, activation='relu')(h)

    h_raster = layers.Dense(canvas_width * canvas_width // 4)(h)
    h_raster = layers.BatchNormalization()(h_raster)
    h_raster = layers.Activation('relu')(h_raster)
    h_raster = layers.Reshape(
        [canvas_width // 8, canvas_width // 8, 16])(h_raster)
    y_raster = rasterNet(h_raster)

    h_shade = layers.Dense(canvas_width * canvas_width // 4)(h)
    h_shade = layers.BatchNormalization()(h_shade)
    h_shade = layers.Activation('relu')(h_shade)
    h_shade = layers.Reshape(
        [canvas_width // 8, canvas_width // 8, 16])(h_shade)
    y_shade = shadeNet(h_shade)

    model = RenderNet(inputs=x, outputs=[y_raster, y_shade])
    return model


class RenderNet(tf.keras.Model):

    def render(self, params):
        raster, shade = self(params)
        stroke = tf.where(raster > 0, 1, 0) * shade
        return stroke

    def compile(self, optimizer, **kwargs):
        kwargs['loss'] = [tf.keras.losses.BinaryCrossentropy(), PixelLoss()]
        super(RenderNet, self).compile(optimizer, **kwargs)

    def save(self, filepath, **kwargs):
        kwargs['include_optimizer'] = False
        super(RenderNet, self).save(filepath, **kwargs)


tf.keras.utils.get_custom_objects().update(
    {
        'PixelShuffle': PixelShuffle,
        'RenderNet': RenderNet
    }
)


if __name__ == '__main__':
    model = renderNet(9, 128)
    model.save('model.h5')
    model = tf.keras.models.load_model('model.h5', compile=False)
    model.summary()
