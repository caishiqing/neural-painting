import tensorflow as tf
import numpy as np


class Renderer(object):
    shape_size = None
    color_size = None
    shape_params = None
    color_params = None
    alpha = None
    foreground = None
    stroke_alpha_map = None

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
    def params(self):
        assert self.shape_params is not None and self.color_params is not None and self.alpha is not None
        return np.concatenate([self.shape_params, self.color_params, [self.alpha]])

    @property
    def param_size(self):
        assert self.shape_size is not None and self.color_size is not None
        return self.shape_size + self.color_size + 1

    def random_params(self):
        self.shape_params = np.random.uniform(0, 1, size=self.shape_size)
        self.color_params = np.random.uniform(0, 1, size=self.color_size)
        self.alpha = np.random.uniform(0, 1)

    def _update_canvas(self):
        self.canvas = self.foreground * self.stroke_alpha_map + self.canvas * (1 - self.stroke_alpha_map)

    def _scaling(self, x):
        return (int)(x * (self.canvas_width - 1) + 0.5)

    def draw_stroke(self):
        raise NotImplemented

    def generate_dataset(self, batch_size=64):
        assert self.shape_size is not None and self.color_size is not None
        def _gen():
            while True:
                self.random_params()
                self.draw_stroke()
                yield self.params, self.foreground

        dataset = tf.data.Dataset.from_generator(
            _gen, output_types=(tf.float32, tf.float32),
            output_shapes=((self.param_size,), (self.canvas_width, self.canvas_width, 3))
        ).batch(batch_size)

        return dataset
