from typing import Any

from challenges.base_model import MobileNetV3Model


class GuessAge:
    def __init__(self) -> None:
        self.model = MobileNetV3Model(num_classes=104)
        self.model.load_model("resources/model_age.h5")

    def __call__(self, image, *args: Any, **kwds: Any) -> Any:
        return self.model.predict([image])
