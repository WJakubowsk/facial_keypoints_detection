import numpy as np
from torchvision.utils import _log_api_usage_once
from typing import Dict
import cv2

class ImageBrightness(object):
    def __init__(self, brightness: float, probability_threshold: float = 0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.probability_threshold = probability_threshold
        self.brightness = brightness

    def __call__(self, image_with_keypoints: Dict[np.array, np.array]) -> Dict[np.array, np.array]:
        if np.random.random() < self.probability_threshold:
            image, keypoints = image_with_keypoints["image"], image_with_keypoints["keypoints"]
            hsv = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            lim = 255 - self.brightness
            v[v > lim] = 255
            v[v <= lim] += self.brightness
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            brightened_image = np.clip(img, 0.0, 1.0)
            return {'image': brightened_image, 'keypoints': keypoints}

        return image_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability_threshold})"