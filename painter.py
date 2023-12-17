from renderer import Renderer
from model import RenderNet
import tensorflow as tf
import numpy as np


class Painter(object):

    def __init__(self,
                 renderer: Renderer,
                 model: RenderNet):

        assert renderer.canvas.shape == model.output_shape[1][1:]
        self.renderer = renderer
        self.model = model
        self.model.trainable = False

        if self.renderer.canvas_color == 'white':
            init_canvas = tf.ones(self.renderer.canvas.shape)
        else:
            init_canvas = tf.zeros(self.renderer.canvas.shape)

        self.init_canvas = tf.expand_dims(init_canvas, 0)

    def _step(self, params, states):
        canvas = states[0]
        raster, shade = self.model(params)
        canvas = raster * shade + canvas * (1 - raster)
        return canvas, [canvas]

    @tf.function
    def _infer(self, params_sequence: tf.Tensor):
        canvas, _, _ = tf.keras.backend.rnn(self._step,
                                            params_sequence,
                                            [self.init_canvas])
        return canvas

    def optimize(self,
                 ground_truth: np.ndarray,
                 num_strokes: int = 20,
                 num_steps: int = 100,
                 learning_rate: float = 1e-3):

        num_samples = ground_truth.shape[0]
        ground_truth = tf.convert_to_tensor(ground_truth)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        init_params = np.random.uniform(0, 1, size=(num_samples,
                                                    num_strokes,
                                                    self.renderer.param_size))
        stroke_params = tf.Variable(initial_value=init_params,
                                    trainable=True,
                                    dtype=tf.float32,
                                    constraint=tf.keras.constraints.MinMaxNorm(0, 1))

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                prediction = self._infer(stroke_params)
                loss = tf.keras.losses.mae(ground_truth, prediction)
                loss = tf.reduce_mean(loss)
            print(loss.numpy())
            optimizer.minimize(loss, [stroke_params], tape=tape)

        return stroke_params.numpy()


if __name__ == '__main__':
    from renderer import OilBrush
    import tensorflow as tf
    from matplotlib import image as mpimg
    import numpy as np

    painter = Painter(renderer=OilBrush(256, canvas_color='black'),
                      model=tf.keras.models.load_model('models/oilbrush-256-white.h5'))

    image = mpimg.imread('lisa.jpg')[:256, :256, :]
    image = image.astype(np.float32)[None, :, :, :] / 255
    params = painter.optimize(image)
