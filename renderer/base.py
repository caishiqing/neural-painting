import numpy as np
import cv2


class Renderer(object):
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

        self.shape_params = None
        self.color_params = None
        self.alpha = None
        self.canvas = None
        self.foreground = None
        self.stroke_alpha_map = None

    @property
    def params(self):
        assert self.shape_params is not None and self.color_params is not None and self.alpha is not None
        return self.shape_params + self.color_params + [self.alpha]

    @property
    def param_size(self):
        assert self.shape_params is not None and self.color_params is not None and self.alpha is not None
        return len(self.shape_params) + len(self.color_params) + 1

    def random_params(self):
        return np.random.uniform(0, 1, size=self.param_size).astype(np.float32)

    def _update_canvas(self):
        self.canvas = self.foreground * self.stroke_alpha_map + self.canvas * (1 - self.stroke_alpha_map)

    def draw_stroke(self):
        raise NotImplemented
