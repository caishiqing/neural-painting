from .base import Renderer
import numpy as np
import utils
import cv2


class Markerpen(Renderer):
    d_shape = 8
    d_color = 3
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
        alpha = self._random_uniform(0.8, 0.98, 1)
        self.stroke_params = np.array(x + r + color + alpha, dtype=np.float32)

    def check_stroke(self):
        r_ = max(self.stroke_params[6], self.stroke_params[7])
        if r_ > 0.025:
            return True
        return False

    def draw_stroke(self):
        # x0, y0, x1, y1, x2, y2, radius0, radius2, R, G, B, A
        x0, y0, x1, y1, x2, y2, radius, _ = self.stroke_params[0:8]
        R0, G0, B0, ALPHA = self.stroke_params[8:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = self._normalize(x0)
        x1 = self._normalize(x1)
        x2 = self._normalize(x2)
        y0 = self._normalize(y0)
        y1 = self._normalize(y1)
        y2 = self._normalize(y2)
        radius = (int)(1 + radius * self.canvas_size // 4)

        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8)  # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8)  # uint8 for antialiasing

        if abs(x0-x2) + abs(y0-y2) < 4:  # too small, dont draw
            self.foreground = np.array(self.foreground, dtype=np.float32) / 255.
            self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32) / 255.
            self.canvas = self._update_canvas()
            return

        color = (R0 * 255, G0 * 255, B0 * 255)
        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        tmp = 1. / 100
        for i in range(100):
            t = i * tmp
            x = (1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2
            y = (1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2

            ptc = (x, y)
            dx = 2 * (t - 1) * x0 + 2 * (1 - 2 * t) * x1 + 2 * t * x2
            dy = 2 * (t - 1) * y0 + 2 * (1 - 2 * t) * y1 + 2 * t * y2

            theta = np.arctan2(dx, dy) - np.pi/2
            pt0 = utils.rotate_pt(pt=(x - radius, y - radius), rotate_center=ptc, theta=theta)
            pt1 = utils.rotate_pt(pt=(x + radius, y - radius), rotate_center=ptc, theta=theta)
            pt2 = utils.rotate_pt(pt=(x + radius, y + radius), rotate_center=ptc, theta=theta)
            pt3 = utils.rotate_pt(pt=(x - radius, y + radius), rotate_center=ptc, theta=theta)
            ppt = np.array([pt0, pt1, pt2, pt3], np.int32)
            ppt = ppt.reshape((-1, 1, 2))
            cv2.fillPoly(self.foreground, [ppt], color, lineType=cv2.LINE_AA)
            cv2.fillPoly(self.stroke_alpha_map, [ppt], alpha, lineType=cv2.LINE_AA)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()
