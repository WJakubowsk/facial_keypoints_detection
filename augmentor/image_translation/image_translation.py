import numpy as np
from torchvision.utils import _log_api_usage_once
from typing import Dict, Tuple
import cv2

class ImageTranslation(object):
    def __init__(self, translation_params: Tuple[float, float], probability_threshold: float = 0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.probability_threshold = probability_threshold
        self.translation_params = translation_params

    def __call__(self, image_with_keypoints: Dict[np.array, np.array]) -> Dict[np.array, np.array]:
        if np.random.random() < self.probability_threshold:
            image, keypoints = image_with_keypoints["image"], image_with_keypoints["keypoints"]
            image_width, image_height = image.shape[0], image.shape[1]
            x_translate_rate, y_translate_rate = self.translation_params
            x_translate_pixel = image_width * x_translate_rate        
            y_translate_pixel = image_height * y_translate_rate

            # image translation
            translation_matrix = np.float32([[1, 0, x_translate_pixel], [0, 1, y_translate_pixel]])
            translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
            
            # keypoints translation
            translated_keypoints = np.copy(keypoints)
            for i in range(len(keypoints)//2):
                translated_keypoints[2 * i] = keypoints[2 * i] + x_translate_pixel
                translated_keypoints[2 * i + 1] = keypoints[2 * i + 1] + y_translate_pixel
            return {'image': translated_image, 'keypoints': translated_keypoints}

        return image_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability_threshold})"