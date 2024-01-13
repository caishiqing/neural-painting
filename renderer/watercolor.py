from .base import Renderer
import numpy as np
import cv2


class Watercolor(Renderer):
    d_shape = 8
    d_color = 6
    d_alpha = 1
    x_index = np.array([0, 4])
    y_index = np.array([1, 5])
    r_index = np.array([6, 7])

    def random_stroke_params_sampler(self, err_map, img):
        cx, cy = self._random_sample_xy(err_map)
        map_h, map_w, _ = img.shape
        x0, y0, x1, y1, x2, y2 = cx, cy, cx, cy, cx, cy
        x = [x0, y0, x1, y1, x2, y2]
        r = self._random_uniform(0.1, 0.25, 2)
        color = img[int(cy*map_h), int(cx*map_w), :].tolist()
        color = color + color
        alpha = self._random_uniform(0.98, 1.0, 1)
        self.stroke_params = np.array(x + r + color + alpha, dtype=np.float32)

    def check_stroke(self):
        r_ = r_ = max(self.stroke_params[6], self.stroke_params[7])
        if r_ > 0.025:
            return True
        return False

    def draw_stroke(self):
        # x0, y0, x1, y1, x2, y2, radius0, radius2, R0, G0, B0, R2, G2, B2, A
        x0, y0, x1, y1, x2, y2, radius0, radius2 = self.stroke_params[0:8]
        R0, G0, B0, R2, G2, B2, ALPHA = self.stroke_params[8:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = self._normalize(x0)
        x1 = self._normalize(x1)
        x2 = self._normalize(x2)
        y0 = self._normalize(y0)
        y1 = self._normalize(y1)
        y2 = self._normalize(y2)
        radius0 = (int)(1 + radius0 * self.canvas_size // 4)
        radius2 = (int)(1 + radius2 * self.canvas_size // 4)

        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8)  # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8)  # uint8 for antialiasing

        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        tmp = 1. / 100
        for i in range(100):
            t = i * tmp
            x = (int)((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
            y = (int)((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
            radius = (int)((1 - t) * radius0 + t * radius2)
            color = ((1-t)*R0*255 + t*R2*255,
                     (1-t)*G0*255 + t*G2*255,
                     (1-t)*B0*255 + t*B2*255)
            cv2.circle(self.foreground, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(self.stroke_alpha_map, (x, y), radius, alpha, -1, lineType=cv2.LINE_AA)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()
