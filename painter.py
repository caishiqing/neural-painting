from matplotlib import pyplot as plt
from renderer import RENDER_MAP
from networks import define_G
from tqdm import tqdm
import numpy as np
import morphology
import logging
import random
import utils
import torch
import loss
import cv2
import os

logging.basicConfig(level=logging.DEBUG)


class Painter(object):
    def __init__(self,
                 model_path: str,
                 img_path: str,
                 render_type: str = "oilpaintbrush",
                 canvas_size: int = 512,
                 canvas_color: str = "white",
                 m_grid: int = 1,
                 max_divide: int = 1,
                 max_m_strokes: int = 1000,
                 model_type: str = "zou-fusion-net-light",
                 output_dir: str = "output",
                 keep_aspect_ratio: bool = False,
                 beta_L1: float = 1.0,
                 beta_ot: float = 0,
                 disable_preview: bool = False,
                 lr: float = 1e-3):

        self.canvas_size = canvas_size
        self.max_divide = max_divide
        self.m_grid = m_grid
        self.max_m_strokes = max_m_strokes
        self.img_path = img_path
        self.output_dir = output_dir
        self.keep_aspect_ratio = keep_aspect_ratio
        self.beta_L1 = beta_L1
        self.beta_ot = beta_ot
        self.disable_preview = disable_preview
        self.lr = lr

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(name="painter")
        self.rderr = RENDER_MAP[render_type](canvas_size=canvas_size,
                                             canvas_color=canvas_color)

        # Define and load model
        self.logger.info('loading renderer from pre-trained checkpoint...')
        self.net_G = define_G(rdrr=self.rderr, netG=model_type).to(self.device)
        checkpoint = torch.load(model_path, self.device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(self.device).eval()

        # define some other vars to record the training states
        self.params = None

        self.G_pred_foreground = None
        self.G_pred_alpha = None
        self.G_final_pred_canvas = torch.zeros(
            [1, 3, self.net_G.out_size, self.net_G.out_size]).to(self.device)

        self.G_loss = torch.tensor(0.0).to(self.device)
        self.step_id = 0
        self.anchor_id = 0

        # define the loss functions
        self._pxl_loss = loss.PixelLoss(p=1)
        self._sinkhorn_loss = loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=False)

        self.target_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.target_image = cv2.cvtColor(
            self.target_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.input_aspect_ratio = self.target_image.shape[0] / self.target_image.shape[1]
        self.target_image = cv2.resize(self.target_image,
                                       (self.net_G.out_size * max_divide,
                                        self.net_G.out_size * max_divide),
                                       cv2.INTER_AREA)

        self.img_batch = utils.img2patches(
            self.target_image, self.m_grid, self.net_G.out_size).to(self.device)

        self.final_rendered_images = None

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    @property
    def num_strokes_per_block(self):
        total_blocks = 0
        for i in range(self.m_grid, self.max_divide + 1):
            total_blocks += i ** 2

        return int(self.max_m_strokes / total_blocks)

    @property
    def psnr(self):
        target = self.img_batch.detach()
        canvas = self.G_pred_canvas.detach()
        return utils.cpt_batch_psnr(canvas, target, PIXEL_MAX=1.0)

    def _save_stroke_params(self, params):
        self.logger.info('saving stroke parameters...')
        file_name = os.path.join(self.output_dir, self.img_path.split('/')[-1][:-4])
        np.savez(file_name + '_strokes.npz', params=params)

    def _shuffle_strokes_and_reshape(self, params):
        grid_idx = list(range(self.m_grid ** 2))
        random.shuffle(grid_idx)
        params = params[grid_idx, :, :]
        params = np.reshape(np.transpose(params, [1, 0, 2]), [-1, self.rderr.param_size])
        params = np.expand_dims(params, axis=0)

        return params

    def _render(self, params, save_jpgs=True, save_video=True):
        params = params[0, :, :]
        if self.keep_aspect_ratio:
            if self.input_aspect_ratio < 1:
                out_h = int(self.canvas_size * self.input_aspect_ratio)
                out_w = self.canvas_size
            else:
                out_h = self.canvas_size
                out_w = int(self.canvas_size / self.input_aspect_ratio)
        else:
            out_h = self.canvas_size
            out_w = self.canvas_size

        file_name = os.path.join(self.output_dir, self.img_path.split('/')[-1][:-4])
        if save_video:
            video_writer = cv2.VideoWriter(file_name + '_animated.mp4',
                                           cv2.VideoWriter_fourcc(*'MP4V'), 40,
                                           (out_w, out_h))

        self.logger.info('rendering canvas...')
        self.rderr.create_empty_canvas()
        for i in range(params.shape[0]):  # for each stroke
            self.rderr.stroke_params = params[i, :]
            if self.rderr.check_stroke():
                self.rderr.draw_stroke()
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            # if save_jpgs:
            #     plt.imsave(file_name + '_rendered_stroke_' + str((i+1)).zfill(4) +
            #                '.png', this_frame)
            if save_video:
                video_writer.write(
                    (this_frame[:, :, ::-1] * 255.).astype(np.uint8))

        if save_jpgs:
            self.logger.info('saving input photo...')
            out_img = cv2.resize(self.target_image, (out_w, out_h), cv2.INTER_AREA)
            plt.imsave(file_name + '_input.png', out_img)

        final_rendered_image = np.copy(this_frame)
        if save_jpgs:
            self.logger.info('saving final rendered result...')
            plt.imsave(file_name + '_final.png', final_rendered_image)

        return final_rendered_image

    def _normalize_strokes(self, params):
        params = np.array(params.detach().cpu())
        for y_id in range(self.m_grid):
            for x_id in range(self.m_grid):
                y_bias = y_id / self.m_grid
                x_bias = x_id / self.m_grid
                params[y_id * self.m_grid + x_id, :, self.rderr.y_index] = \
                    y_bias + params[y_id * self.m_grid + x_id, :, self.rderr.y_index] / self.m_grid
                params[y_id * self.m_grid + x_id, :, self.rderr.x_index] = \
                    x_bias + params[y_id * self.m_grid + x_id, :, self.rderr.x_index] / self.m_grid
                params[y_id * self.m_grid + x_id, :, self.rderr.r_index] /= self.m_grid

        return params

    def initialize_params(self):
        self.params = torch.rand((self.m_grid*self.m_grid, self.num_strokes_per_block, self.rderr.param_size),
                                 dtype=torch.float32, device=self.device)
        self.params.requires_grad = True

    def stroke_sampler(self, anchor_id):
        if anchor_id == self.num_strokes_per_block:
            return

        err_maps = torch.sum(torch.abs(self.img_batch - self.G_final_pred_canvas),
                             dim=1, keepdim=True).detach()

        for i in range(self.m_grid*self.m_grid):
            this_err_map = err_maps[i, 0, :, :].cpu().numpy()
            ks = int(this_err_map.shape[0] / 8)
            this_err_map = cv2.blur(this_err_map, (ks, ks))
            this_err_map = this_err_map ** 4
            this_img = self.img_batch[i, :, :, :].detach().permute([1, 2, 0]).cpu().numpy()
            self.rderr.random_stroke_params_sampler(err_map=this_err_map, img=this_img)
            params = torch.tensor(self.rderr.stroke_params)
            self.params.data[i, anchor_id, :] = params

    def _backward_x(self):
        self.G_loss = 0
        self.G_loss += self.beta_L1 * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.img_batch)
        if self.beta_ot > 0:
            self.G_loss += self.beta_ot * self._sinkhorn_loss(
                self.G_final_pred_canvas, self.img_batch)
        self.G_loss.backward()

    def _forward_pass(self):
        params = torch.reshape(self.params[:, 0:self.anchor_id+1, :],
                               [self.m_grid*self.m_grid*(self.anchor_id+1), -1, 1, 1])

        self.G_pred_foregrounds, self.G_pred_alphas = self.net_G(params)

        self.G_pred_foregrounds = morphology.Dilation2d(m=1)(self.G_pred_foregrounds)
        self.G_pred_alphas = morphology.Erosion2d(m=1)(self.G_pred_alphas)

        self.G_pred_foregrounds = torch.reshape(
            self.G_pred_foregrounds, [self.m_grid*self.m_grid, self.anchor_id+1, 3,
                                      self.net_G.out_size, self.net_G.out_size])
        self.G_pred_alphas = torch.reshape(
            self.G_pred_alphas, [self.m_grid*self.m_grid, self.anchor_id+1, 3,
                                 self.net_G.out_size, self.net_G.out_size])

        for i in range(self.anchor_id+1):
            G_pred_foreground = self.G_pred_foregrounds[:, i]
            G_pred_alpha = self.G_pred_alphas[:, i]
            self.G_pred_canvas = G_pred_foreground * G_pred_alpha \
                + self.G_pred_canvas * (1 - G_pred_alpha)

        self.G_final_pred_canvas = self.G_pred_canvas

    def _drawing_step_states(self):
        vis2 = utils.patches2img(
            self.G_final_pred_canvas, self.m_grid).clip(min=0, max=1)
        if self.disable_preview:
            pass
        else:
            cv2.namedWindow('G_pred', cv2.WINDOW_NORMAL)
            cv2.namedWindow('input', cv2.WINDOW_NORMAL)
            cv2.imshow('G_pred', vis2[:, :, ::-1])
            cv2.imshow('input', self.target_image[:, :, ::-1])
            cv2.waitKey(1)

    def optimize(self):
        self.logger.info('begin drawing...')
        PARAMS = np.zeros([1, 0, self.rderr.param_size], np.float32)

        if self.rderr.canvas_color == 'white':
            canvas = torch.ones([1, 3, self.net_G.out_size, self.net_G.out_size]).to(self.device)
        else:
            canvas = torch.zeros([1, 3, self.net_G.out_size, self.net_G.out_size]).to(self.device)

        for self.m_grid in range(self.m_grid, self.max_divide + 1):
            # for each scale level
            self.img_batch = utils.img2patches(self.target_image, self.m_grid, self.net_G.out_size).to(self.device)
            self.G_final_pred_canvas = canvas

            self.initialize_params()
            utils.set_requires_grad(self.net_G, False)
            optimizer = torch.optim.RMSprop([self.params], lr=self.lr, centered=True)

            self.step_id = 0
            iters_per_stroke = int(500 / self.num_strokes_per_block)
            for self.anchor_id in range(0, self.num_strokes_per_block):
                self.stroke_sampler(self.anchor_id)
                pbar = tqdm(range(iters_per_stroke))
                for _ in pbar:
                    self.G_pred_canvas = canvas

                    # update x
                    optimizer.zero_grad()
                    self._forward_pass()
                    self._drawing_step_states()
                    self._backward_x()

                    pbar.set_description('iteration step %d, G_loss: %.5f, psnr: %.5f, grid_scale: %d / %d, strokes: %d / %d'
                                         % (self.step_id, self.G_loss.item(), self.psnr.item(), self.m_grid, self.max_divide,
                                            self.anchor_id + 1, self.num_strokes_per_block))

                    self.params.data = torch.clamp(self.params.data, 0, 1)
                    self.params.data[:, :, :self.rderr.d_shape] = torch.clamp(self.params.data[:, :, :self.rderr.d_shape], 0.1, 0.9)

                    optimizer.step()
                    self.step_id += 1

            v = self._normalize_strokes(self.params)
            v = self._shuffle_strokes_and_reshape(v)
            PARAMS = np.concatenate([PARAMS, v], axis=1)
            canvas = self._render(PARAMS, save_jpgs=False, save_video=False)
            canvas = utils.img2patches(canvas, self.m_grid + 1, self.net_G.out_size).to(self.device)

        self._save_stroke_params(PARAMS)
        self._render(PARAMS, save_jpgs=True, save_video=True)
