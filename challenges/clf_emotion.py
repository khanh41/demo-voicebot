from typing import Any

import cv2
import numpy as np

from challenges.base_model import MobileNetV3Model
from challenges.data_loader import load_image_from_directory
from constants import Chatbot


class EmotionDetect:
    def __init__(self) -> None:
        self.model = MobileNetV3Model(num_classes=7)
        self.model.load_model("resources/model_emotion.h5")

    def __call__(self, image, *args: Any, **kwds: Any) -> Any:
        image = cv2.resize(image, (224, 224))
        image = image.reshape((1, 224, 224, 3))
        result = self.model.predict(image)
        return Chatbot.emotions[np.argmax(result)]


if __name__ == '__main__':
    train_data, val_ds, class_names = load_image_from_directory("train", 0.1)

    model = MobileNetV3Model(num_classes=7)
    model.create_model()
    model.compile()

    model.fit(train_data, 10, val_ds)
    model.save_model("model.h5")
