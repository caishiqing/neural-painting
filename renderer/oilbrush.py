from .base import Renderer
import numpy as np
import cv2


class OilBrush(Renderer):
    # params: [xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2]
    shape_size = 5
    color_size = 6
    brush_small_vertical = cv2.imread(
        r'./brushes/brush_fromweb2_small_vertical.png', cv2.IMREAD_GRAYSCALE)
    brush_small_horizontal = cv2.imread(
        r'./brushes/brush_fromweb2_small_horizontal.png', cv2.IMREAD_GRAYSCALE)
    brush_large_vertical = cv2.imread(
        r'./brushes/brush_fromweb2_large_vertical.png', cv2.IMREAD_GRAYSCALE)
    brush_large_horizontal = cv2.imread(
        r'./brushes/brush_fromweb2_large_horizontal.png', cv2.IMREAD_GRAYSCALE)

    def _process_params(self, params):
        return np.concatenate([self._scaling(params[:4]), [params[4] * np.math.pi], params[5:]])

    def draw_stroke(self, params: np.ndarray):
        params = self._process_params(params)
        w, h = params[2:4]
        if w * h / (self.canvas_width**2) > 0.1:
            if h > w:
                brush = self.brush_large_vertical
            else:
                brush = self.brush_large_horizontal
        else:
            if h > w:
                brush = self.brush_small_vertical
            else:
                brush = self.brush_small_horizontal

        alpha, brush = self.create_transformed_brush(brush,
                                                     self.canvas_width,
                                                     self.canvas_width,
                                                     params)

        if not self.train:
            alpha = cv2.dilate(alpha, np.ones([2, 2]))
            brush = cv2.dilate(brush, np.ones([2, 2]))

        if len(alpha.shape) == 2:
            alpha = np.expand_dims(alpha, -1)

        return alpha, brush

    def create_transformed_brush(self, brush, canvas_w, canvas_h, params):
        shape_params, color_params, _ = self._parse_params(params)
        alpha = (brush > 0).astype(np.float32)[:, :, None]

        t = np.linspace(0, 1, brush.shape[0])[:, None]
        color = (1 - t) * color_params[:3][None, :] + t * color_params[3:][None, :]
        colormap = np.expand_dims(color, 1)

        brush = brush.astype(np.float32)[:, :, None] / 255
        brush = brush * colormap

        M1 = build_transformation_matrix([-brush.shape[1]/2, -brush.shape[0]/2, 0])
        M2 = build_scale_matrix(
            sx=shape_params[2]/brush.shape[1], sy=shape_params[3]/brush.shape[0])
        M3 = build_transformation_matrix([0, 0, shape_params[4]])
        M4 = build_transformation_matrix([shape_params[0], shape_params[1], 0])

        M = update_transformation_matrix(M1, M2)
        M = update_transformation_matrix(M, M3)
        M = update_transformation_matrix(M, M4)

        alpha = cv2.warpAffine(
            alpha, M, (canvas_w, canvas_h),
            borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)

        brush = cv2.warpAffine(
            brush, M, (canvas_w, canvas_h),
            borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)

        return alpha, brush


def build_scale_matrix(sx, sy):
    transform_matrix = np.zeros((2, 3))
    transform_matrix[0, 0] = sx
    transform_matrix[1, 1] = sy
    return transform_matrix


def update_transformation_matrix(M, m):

    # extend M and m to 3x3 by adding an [0,0,1] to their 3rd row
    M_ = np.concatenate([M, np.zeros([1, 3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1, 3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]


def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix
    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix
