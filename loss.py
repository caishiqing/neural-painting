import tensorflow as tf


class PixelLoss(tf.keras.losses.MeanAbsoluteError):

    def call(self, gt, canvas):
        mask = tf.where(tf.reduce_max(gt, -1) > 0, True, False)
        gt = tf.boolean_mask(gt, mask)
        canvas = tf.boolean_mask(canvas, mask)
        return super(PixelLoss, self).call(gt, canvas)


class SinkhornLoss(tf.keras.layers.Layer):

    def __init__(self,
                 ground_truth: tf.Tensor,
                 resize_width: int = None,
                 lamb: float = 0.1,
                 **kwargs):

        super(SinkhornLoss, self).__init__(**kwargs)
        self.resize_width = resize_width
        self.lamb = lamb
        self.ground_truth = tf.convert_to_tensor(ground_truth)

        if resize_width is not None:
            self.ground_truth = tf.image.resize(self.ground_truth,
                                                (resize_width, resize_width))

        self.ground_truth = tf.keras.layers.Flatten()(self.ground_truth)
        self.batch_size, self.gt_size = tf.keras.backend.int_shape(self.ground_truth)

    def build(self, input_shape):
        b, h, w, c = input_shape
        assert b == self.batch_size
        if self.resize_width is not None:
            h = w = self.resize_width

        self.P = self.add_weight(name="transport",
                                 shape=(self.batch_size, h * w, self.gt_size),
                                 dtype=tf.float32,
                                 initializer='glorot_uniform',
                                 constraint=tf.keras.constraints.NonNeg,
                                 trainable=True)

        self.constraint_loss = tf.keras.losses.MAE
        self.built = True

    def call(self, predictions):
        if self.resize_width is not None:
            predictions = tf.image.resize(predictions, (self.resize_width, self.resize_width))

        predictions = tf.keras.layers.Flatten()(predictions)
        distance_matrix = tf.abs(predictions[:, :, tf.newaxis] - self.ground_truth[:, tf.newaxis, :])
        cost_loss = tf.reduce_mean(self.P * distance_matrix, axis=[1, 2])
        row_loss = self.constraint_loss(tf.reduce_sum(self.P, axis=2) - predictions)
        col_loss = self.constraint_loss(tf.reduce_sum(self.P, axis=1) - self.ground_truth)

        loss = tf.reduce_mean(cost_loss + row_loss + col_loss)
        return loss


class MyLoss(tf.keras.layers.Layer):
    def __init__(self, ground_truth, **kwargs):
        super(MyLoss, self).__init__(**kwargs)
        self.ground_truth = tf.convert_to_tensor(ground_truth)
        self._loss = tf.keras.losses.mae

    def call(self, predictions):
        loss = self._loss(self.ground_truth, predictions)
        return loss
