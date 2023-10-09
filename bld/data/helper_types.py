from typing import NamedTuple, Dict, Optional, Tuple

BoundingBox = Tuple[float, float, float, float]

class Annotation(NamedTuple):
    area: float
    image_id: str
    bbox: BoundingBox
    category_no: int
    category_id: str
    id: Optional[int] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    is_group_of: Optional[bool] = None
    is_truncated: Optional[bool] = None
    is_occluded: Optional[bool] = None
    is_depiction: Optional[bool] = None
    is_inside: Optional[bool] = None
    segmentation: Optional[Dict] = None