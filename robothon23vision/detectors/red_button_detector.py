from dataclasses import dataclass
import cv2
import numpy as np

from robothon23vision.detectors.feature_detector import FeatureDetector
from robothon23vision.detectors.errors import NoCircleDetectedError, TooManyCirclesDetectedError


@dataclass
class RedButtonDetectorConfig:
    min_radius: float = 9
    max_radius: float = 15
    circle_param_1: float = 20
    circle_param_2: float = 13
    kernel: bool = False
    h_low: int = 150
    s_low: int = 80
    v_low: int = 100
    h_up: int = 240
    s_up: int = 200
    v_up: int = 255

class RedButtonDetector(FeatureDetector):
    def __init__(self, img_in, **config):
        super().__init__(img_in, 'red_button')
        self._config = RedButtonDetectorConfig(**config)

    def set_mask(self) -> None:
        self._hsv = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)

        mask_0 = cv2.inRange(
                self._hsv,
                np.array([self._config.h_low, self._config.s_low, self._config.v_low]),
                np.array([self._config.h_up, self._config.s_up, self._config.v_up]),
                )
        self._mask = mask_0

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
            raise NoCircleDetectedError("No red button detected.")
        else:
            if not circles.shape == (1, 2, 3):
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
                raise TooManyCirclesDetectedError("Too many red circles detected.")
            else:
                circles = np.uint16(np.around(circles))
                # Selecting logic if there are too circles use highest y coordinate
                circle_0 = circles[0, 0, :]
                circle_1 = circles[0, 1, :]

                red_button = (
                        int(0.5 * circle_0[0] + 0.5 * circle_1[0] ),
                        int(0.5 * circle_0[1] + 0.5 * circle_1[1]),
                        int(0.5 * circle_0[2] + 0.5 * circle_1[2])
                        )
                cv2.circle(
                        self._img,
                        red_button[0:2],
                        red_button[2],
                         (0, 255, 255),
                         2)
                cv2.circle(self._img, red_button[0:2], 2, (0, 255, 255), 3)

                results = {
                        self._name:  red_button[0:2],
                        }
                return results

