import tensorflow as tf
import numpy as np


class Renderer(object):
    shape_size = None
    color_size = None

    def __init__(self, canvas_width=128, canvas_color='white', train=False):
        self.canvas_width = canvas_width
        self.canvas_colour = canvas_color
        self.train = train
        if canvas_color == 'white':
            self.canvas = np.ones((canvas_width, canvas_width, 3), dtype=np.float32)
        elif canvas_color == 'black':
            self.canvas = np.zeros((canvas_width, canvas_width, 3), dtype=np.float32)
        else:
            raise Exception(f"Canvas colour '{canvas_color}' is not supported!")

    @property
    def param_size(self):
        assert self.shape_size is not None and self.color_size is not None
        return self.shape_size + self.color_size

    def random_params(self):
        return np.random.uniform(0, 1, size=self.param_size)

    def _update_canvas(self, foreground, stroke_alpha_map):
        self.canvas = foreground * stroke_alpha_map + self.canvas * (1 - stroke_alpha_map)

    def _scaling(self, x):
        return x * (self.canvas_width - 1) + 0.5

    def _parse_params(self, params):
        shape_params = params[:self.shape_size]
        color_params = params[self.shape_size:self.shape_size+self.color_size]
        alpha = params[-1]
        return shape_params, color_params, alpha

    def draw_stroke(self, params: np.ndarray) -> np.ndarray:
        # from stroke params to stroke foreground
        raise NotImplemented

    def update_canvas(self, params):
        # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        foreground, stroke_alpha_map = self._draw_stroke(params)
        self.canvas = foreground * stroke_alpha_map + self.canvas * (1 - stroke_alpha_map)

    def generate_dataset(self, batch_size=64):
        assert self.shape_size is not None and self.color_size is not None

        def _gen():
            while True:
                yield self.random_params()

        def _render(params):
            foreground, stroke_alpha_map = tf.py_function(self.draw_stroke, [params], Tout=[tf.float32, tf.float32])
            return params, (foreground, stroke_alpha_map)

        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_generator(
            _gen, output_types=tf.float32, output_shapes=(self.param_size,)
        ).map(_render, autotune).batch(batch_size)

        return dataset
