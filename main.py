from painter import Painter
import fire


def paint(img_path: str,
          model_path: str,
          **kwargs):

    painter = Painter(img_path=img_path,
                      model_path=model_path,
                      **kwargs)

    painter.optimize()


if __name__ == "__main__":
    fire.Fire()
