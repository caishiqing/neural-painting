from model import renderNet
from renderer import RENDERER_FACTORY
import tensorflow as tf
import fire
import os


def train_renderer(canvas_width: int = 128,
                   canvas_color: str = 'white',
                   renderer_type: str = 'oilbrush',
                   save_dir: str = 'models',
                   gpu: str = '0',
                   **train_args):

    # environment config
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    assert renderer_type in RENDERER_FACTORY
    renderer = RENDERER_FACTORY[renderer_type](
        canvas_width, canvas_color, True)

    # parse args
    learning_rate = train_args.pop('learning_rate', 1e-3)
    batch_size = train_args.pop('batch_size', 64)
    epochs = train_args.pop('epochs', 100)
    steps_per_epoch = train_args.pop('steps_per_epoch', 1000)

    # dataset and model
    dataset = renderer.generate_dataset(batch_size)
    model = renderNet(renderer.param_size, canvas_width)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    # build callbacks
    schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=steps_per_epoch * epochs,
        end_learning_rate=0.1 * learning_rate
    )
    schedule = tf.keras.callbacks.LearningRateScheduler(schedule)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir,
                              '{}-{}-{}.h5'.format(renderer_type,
                                                   canvas_width,
                                                   canvas_color))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                    monitor='loss',
                                                    save_best_only=True)
    callbacks = [schedule, checkpoint]

    # train model
    model.fit(x=dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              **train_args)


if __name__ == '__main__':
    fire.Fire()
