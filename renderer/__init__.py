from .base import Renderer
from .oilbrush import OilBrush
from .markerpen import Markerpen
from .watercolor import Watercolor
from .rectangle import Rectangle


RENDER_MAP = {
    "oilbrush": OilBrush,
    "markerpen": Markerpen,
    "watercolor": Watercolor,
    "rectangle": Rectangle
}
