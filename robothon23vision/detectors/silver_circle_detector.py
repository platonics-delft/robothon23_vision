from dataclasses import dataclass
import cv2
import numpy as np

from robothon23vision.detectors.feature_detector import FeatureDetector
from robothon23vision.detectors.errors import NoCircleDetectedError, TooManyCirclesDetectedError


@dataclass
class SilverCircleDetectorConfig:
    min_radius: float = 10
    max_radius: float = 15
    circle_param_1: float = 30
    circle_param_2: float = 14
    kernel: bool = False
    h_low: int = 40
    s_low: int = 0
    v_low: int = 250
    h_high: int = 80
    s_high: int = 10
    v_high: int = 256


class SilverCircleDetector(FeatureDetector):
    def __init__(self, img_in, **config):
        super().__init__(img_in, 'silver_circle')
        self._config = SilverCircleDetectorConfig(**config)

    def set_mask(self) -> None:
        self._hsv = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)

        # Detect circles using HoughCircles

        self._mask = cv2.inRange(
                self._hsv,
                np.array([self._config.h_low, self._config.s_low, self._config.v_low]),
                np.array([self._config.h_high, self._config.s_high, self._config.v_high]),
                )

        if self._config.kernel:
            kernel = np.ones((5,5), np.uint8)
            self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_OPEN, kernel)
            self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_CLOSE, kernel)



    def detect(self) -> dict:
        circles = cv2.HoughCircles(
                self._mask,
                cv2.HOUGH_GRADIENT,
                1,
                20,
                param1=self._config.circle_param_1,
                param2=self._config.circle_param_2,
                minRadius=self._config.min_radius,
                maxRadius=self._config.max_radius,
                )

        self._filtered_img = cv2.bitwise_and(self._img, self._img, mask=self._mask)

        if circles is None:
            raise NoCircleDetectedError("No silver circle detected")
        else:
            if not circles.shape == (1, 1, 3):
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
                raise TooManyCirclesDetectedError("Too many silver circles detected.")
            # Draw the outer circle
            silver_circle = (
                    int(circles[:, 0, 0]),
                    int(circles[:, 0, 1]),
                    int(circles[:, 0, 2])
                    )
            cv2.circle(
                    self._img,
                    silver_circle[0:2],
                    silver_circle[2],
                     (0, 255, 0),
                     2)
            cv2.circle(self._img, silver_circle[0:2], 2, (0, 0, 255), 3)

        results = {self._name: silver_circle[0:2]}
        return results

