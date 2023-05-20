from renderer import Renderer
from model import RenderNet
import tensorflow as tf


class Painter(object):

    def __init__(self,
                 renderer: Renderer,
                 model: RenderNet):

        assert renderer.canvas.shape == model.output_shape[1][1:]
        self.renderer = renderer
        self.model = model

    @tf.function
    def _infer(self, params_sequence):
        def _step(params, states):
            canvas = states[0]
            raster, shade = self.model(params)
            canvas = raster * shade + canvas * (1 - raster)
            return canvas, [canvas]

        samples = tf.keras.backend.int_shape(params_sequence)[0]
        if self.renderer.canvas_colour == 'white':
            init_canvas = tf.ones((samples,) + self.renderer.canvas.shape)
        else:
            init_canvas = tf.zeros((samples,) + self.renderer.canvas.shape)

        canvas, _, _ = tf.keras.backend.rnn(_step, params_sequence, [init_canvas])
        return canvas


if __name__ == '__main__':
    from renderer import OilBrush
    oilbrush = OilBrush(canvas_color='black', canvas_width=256)
    model = tf.keras.models.load_model('models/oilbrush-256-white.h5')
    painter = Painter(oilbrush, model)
    params_sequence = tf.random.uniform((1, 10, oilbrush.param_size))
    canvas = painter._infer(params_sequence)
    print(canvas.shape)
