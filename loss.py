import tensorflow as tf


class PixelLoss(tf.keras.losses.MeanAbsoluteError):

    def call(self, gt, canvas):
        mask = tf.where(tf.reduce_max(gt, -1) > 0, True, False)
        gt = tf.boolean_mask(gt, mask)
        canvas = tf.boolean_mask(canvas, mask)
        return super(PixelLoss, self).call(gt, canvas)
