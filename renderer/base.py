import numpy as np
import cv2


class Renderer():
    d_shape = None
    d_color = None
    d_alpha = None
    x_index = None
    y_index = None
    r_index = None
    transparent = True

    def __init__(self,
                 canvas_size=128,
                 canvas_color='black',
                 train=False):

        self.canvas_size = canvas_size
        self.stroke_params = None
        self.canvas_color = canvas_color
        self.train = train
        self.create_empty_canvas()

    def create_empty_canvas(self):
        if self.canvas_color == 'white':
            self.canvas = np.ones(
                [self.canvas_size, self.canvas_size, 3]).astype('float32')
        else:
            self.canvas = np.zeros(
                [self.canvas_size, self.canvas_size, 3]).astype('float32')

    @property
    def param_size(self):
        return self.d_shape + self.d_color + self.d_alpha

    def _random_uniform(self, low, high, size):
        return np.random.uniform(low, high, size).tolist()

    def _update_canvas(self):
        return self.foreground * self.stroke_alpha_map + \
            self.canvas * (1 - self.stroke_alpha_map)

    def _normalize(self, x):
        return (int)(x * (self.canvas_size - 1) + 0.5)

    def _random_sample_xy(self, err_map):
        err_map = cv2.resize(err_map, (self.canvas_size, self.canvas_size))
        err_map[err_map < 0] = 0
        if np.all((err_map == 0)):
            err_map = np.ones_like(err_map)
        err_map = err_map / (np.sum(err_map) + 1e-9)

        index = np.random.choice(range(err_map.size), size=1, p=err_map.ravel())[0]

        cy = (index // self.canvas_size) / self.canvas_size
        cx = (index % self.canvas_size) / self.canvas_size
        return cx, cy

    def random_stroke_params(self):
        self.stroke_params = np.random.uniform(0, 1, self.param_size)

    def random_stroke_params_sampler(self, err_map, img):
        raise NotImplementedError

    def check_stroke(self):
        return True

    def draw_stroke(self):
        raise NotImplementedError
