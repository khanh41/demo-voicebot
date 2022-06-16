from typing import Any

from challenges.base_model import MobileNetV3Model


class EmotionDetect:
    def __init__(self) -> None:
        self.model = MobileNetV3Model(num_classes=7)
        self.model.load_model("resources/model_emotion.h5")

    def __call__(self, image, *args: Any, **kwds: Any) -> Any:
        return self.model.predict([image])
