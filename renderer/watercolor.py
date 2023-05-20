from .base import Renderer
import numpy as np
import cv2


class WaterColor(Renderer):
    shape_size = 8
    color_size = 6

    def _process_params(self, params):
        params[:6] = self._scaling(params[:6])

    def draw_stroke(self, params):
        # x0, y0, x1, y1, x2, y2, radius0, radius2, R0, G0, B0, R2, G2, B2, A
        x0, y0, x1, y1, x2, y2, radius0, radius2 = params[0:8]
        R0, G0, B0, R2, G2, B2, ALPHA = params[8:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = self._scaling(x0)
        x1 = self._scaling(x1)
        x2 = self._scaling(x2)
        y0 = self._scaling(y0)
        y1 = self._scaling(y1)
        y2 = self._scaling(y2)
        radius0 = (int)(1 + radius0 * self.canvas_width // 4)
        radius2 = (int)(1 + radius2 * self.canvas_width // 4)

        stroke_alpha_value = params[-1]

        foreground = np.zeros_like(self.canvas, dtype=np.uint8)
        stroke_alpha_map = np.zeros_like(self.canvas, dtype=np.uint8)

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
            cv2.circle(foreground, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(stroke_alpha_map, (x, y), radius, alpha, -1, lineType=cv2.LINE_AA)

        if not self.train:
            foreground = cv2.dilate(foreground, np.ones([2, 2]))
            stroke_alpha_map = cv2.erode(stroke_alpha_map, np.ones([2, 2]))

        foreground = np.array(foreground, dtype=np.float32)/255.
        stroke_alpha_map = np.array(stroke_alpha_map, dtype=np.float32)/255.

        return stroke_alpha_map, foreground
