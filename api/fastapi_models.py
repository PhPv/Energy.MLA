from typing import Optional

import pydantic


class MetricsRequest(pydantic.BaseModel):
    object_id: str
    model_name: str
    date_range: tuple[str, str] = ()
