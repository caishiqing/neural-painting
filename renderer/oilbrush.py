from .base import Renderer
import numpy as np
import utils
import cv2


class OilBrush(Renderer):
    d_shape = 5
    d_color = 6
    d_alpha = 1
    x_index = np.array([0])
    y_index = np.array([1])
    r_index = np.array([2, 3])
    transparent = False
    brush_small_vertical = cv2.imread(
        r'./brushes/brush_fromweb2_small_vertical.png', cv2.IMREAD_GRAYSCALE)
    brush_small_horizontal = cv2.imread(
        r'./brushes/brush_fromweb2_small_horizontal.png', cv2.IMREAD_GRAYSCALE)
    brush_large_vertical = cv2.imread(
        r'./brushes/brush_fromweb2_large_vertical.png', cv2.IMREAD_GRAYSCALE)
    brush_large_horizontal = cv2.imread(
        r'./brushes/brush_fromweb2_large_horizontal.png', cv2.IMREAD_GRAYSCALE)

    def random_stroke_params_sampler(self, err_map, img):
        cx, cy = self._random_sample_xy(err_map)
        map_h, map_w, _ = img.shape
        x = [cx, cy]
        wh = self._random_uniform(0.1, 0.25, 2)
        theta = self._random_uniform(0, 1, 1)
        color = img[int(cy*map_h), int(cx*map_w), :].tolist()
        color = color + color
        alpha = self._random_uniform(0.98, 1.0, 1)
        self.stroke_params = np.array(x + wh + theta + color + alpha, dtype=np.float32)

    def check_stroke(self):
        r_ = max(self.stroke_params[2], self.stroke_params[3])
        if r_ > 0.025:
            return True
        return False

    def draw_stroke(self):

        # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        x0, y0, w, h, theta = self.stroke_params[0:5]
        R0, G0, B0, R2, G2, B2, ALPHA = self.stroke_params[5:]
        x0 = self._normalize(x0)
        y0 = self._normalize(y0)
        w = (int)(1 + w * self.canvas_size)
        h = (int)(1 + h * self.canvas_size)
        theta = np.pi*theta

        if w * h / (self.canvas_size**2) > 0.1:
            if h > w:
                brush = self.brush_large_vertical
            else:
                brush = self.brush_large_horizontal
        else:
            if h > w:
                brush = self.brush_small_vertical
            else:
                brush = self.brush_small_horizontal
        self.foreground, self.stroke_alpha_map = utils.create_transformed_brush(
            brush, self.canvas_size, self.canvas_size,
            x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()
