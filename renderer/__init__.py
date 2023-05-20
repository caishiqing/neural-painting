from .base import Renderer
from .oilbrush import OilBrush
from .watercolor import WaterColor

RENDERER_FACTORY = {
    'renderer': Renderer,
    'oilbrush': OilBrush,
    'watercolor': WaterColor
}
