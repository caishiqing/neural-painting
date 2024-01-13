from .base import Renderer
import numpy as np
import utils
import cv2


class Rectangle(Renderer):
    d_shape = 5
    d_color = 3
    d_alpha = 1
    x_index = np.array([0])
    y_index = np.array([1])
    r_index = np.array([2, 3])

    def random_stroke_params_sampler(self, err_map, img):
        cx, cy = self._random_sample_xy(err_map)
        map_h, map_w, _ = img.shape
        x = [cx, cy]
        wh = self._random_uniform(0.1, 0.25, 2)
        theta = [0]
        color = img[int(cy*map_h), int(cx*map_w), :].tolist()
        alpha = self._random_uniform(0.8, 0.98, 1)
        self.stroke_params = np.array(x + wh + theta + color + alpha, dtype=np.float32)

    def check_stroke(self):
        r_ = max(self.stroke_params[2], self.stroke_params[3])
        if r_ > 0.025:
            return True
        return False

    def draw_stroke(self):
        # xc, yc, w, h, theta, R, G, B, A
        x0, y0, w, h, theta = self.stroke_params[0:5]
        R0, G0, B0, ALPHA = self.stroke_params[5:]
        x0 = self._normalize(x0)
        y0 = self._normalize(y0)
        w = (int)(1 + w * self.canvas_size // 4)
        h = (int)(1 + h * self.canvas_size // 4)
        theta = np.pi*theta
        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8)  # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8)  # uint8 for antialiasing

        color = (R0 * 255, G0 * 255, B0 * 255)
        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        ptc = (x0, y0)
        pt0 = utils.rotate_pt(pt=(x0 - w, y0 - h), rotate_center=ptc, theta=theta)
        pt1 = utils.rotate_pt(pt=(x0 + w, y0 - h), rotate_center=ptc, theta=theta)
        pt2 = utils.rotate_pt(pt=(x0 + w, y0 + h), rotate_center=ptc, theta=theta)
        pt3 = utils.rotate_pt(pt=(x0 - w, y0 + h), rotate_center=ptc, theta=theta)

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
