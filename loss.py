import tensorflow as tf


class PixelLoss(tf.keras.losses.Loss):
    def __init__(self, power=1, ignore_color=False, **kwargs):
        super(PixelLoss, self).__init__(**kwargs)
        self.power = power
        self.ignore_color = ignore_color

    def call(self, gt, canvas):
        if self.ignore_color:
            gt = tf.reduce_mean(gt, axis=-1)
            canvas = tf.reduce_mean(canvas, axis=-1)

        return tf.pow(tf.abs(gt - canvas), self.power)
