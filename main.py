from model import RenderNet
from renderer import RENDERER_FACTORY
from loss import PixelLoss
import tensorflow as tf
import fire
import os


def train_renderer(canvas_width: int = 128,
                   canvas_color: str = 'white',
                   renderer_type: str = 'oilbrush',
                   save_path: str = None,
                   **train_args):

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    assert renderer_type in RENDERER_FACTORY
    renderer = RENDERER_FACTORY[renderer_type](
        canvas_width, canvas_color, True)

    dataset = renderer.generate_dataset(train_args.pop('batch_size', 64))
    model = RenderNet(renderer.param_size, canvas_width)
    model.compile(optimizer=tf.keras.optimizers.Adam(train_args.pop('learning_rate', 1e-3)),
                  loss=[tf.keras.losses.BinaryCrossentropy(), PixelLoss()]
                  )

    if save_path is not None:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                        monitor='loss',
                                                        save_best_only=True,
                                                        save_weights_only=False)
        callbacks = [checkpoint]
        print(checkpoint)
    else:
        callbacks = None

    model.fit(x=dataset,
              epochs=train_args.pop('epochs', 100),
              steps_per_epoch=train_args.pop('steps_per_epoch', 500),
              callbacks=callbacks,
              verbose=train_args.pop('verbose', 1))


if __name__ == '__main__':
    fire.Fire()
