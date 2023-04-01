from dataclasses import dataclass
import cv2
import numpy as np

from robothon23.detectors.feature_detector import FeatureDetector
from robothon23.detectors.errors import NoCircleDetectedError, TooManyCirclesDetectedError


@dataclass
class RedCirclesDetectorConfig:
    min_radius: float = 3
    max_radius: float = 10
    circle_param_1: float = 20
    circle_param_2: float = 5
    kernel: bool = False
    h_low_0: int = 0
    s_low_0: int = 80
    v_low_0: int = 150
    h_up_0: int = 10
    s_up_0: int = 200
    v_up_0: int = 255
    h_low_1: int = 150
    s_low_1: int = 100
    v_low_1: int = 100
    h_up_1: int = 255
    s_up_1: int = 140
    v_up_1: int = 135

class RedCirclesDetector(FeatureDetector):
    def __init__(self, img_in, **config):
        super().__init__(img_in, 'red_circles')
        self._config = RedCirclesDetectorConfig(**config)

    def set_mask(self) -> None:
        self._hsv = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)

        mask_0 = cv2.inRange(
                self._hsv,
                np.array([self._config.h_low_0, self._config.s_low_0, self._config.v_low_0]),
                np.array([self._config.h_up_0, self._config.s_up_0, self._config.v_up_0]),
                )
        mask_1 = cv2.inRange(
                self._hsv,
                np.array([self._config.h_low_1, self._config.s_low_1, self._config.v_low_1]),
                np.array([self._config.h_up_1, self._config.s_up_1, self._config.v_up_1]),
                )


        self._mask = cv2.bitwise_or(mask_0, mask_1)
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
            raise NoCircleDetectedError("No red circle detected.")
        else:
            if not circles.shape == (1, 2, 3):
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    # Draw the outer circle
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
                raise TooManyCirclesDetectedError("Wrong number of red circles detected.")
                
            # Selecting logic if there are too circles
            if circles[:, 0, 2] > circles[:, 1, 2]:
                red_button_index = 0
                red_circle_index = 1
            else:
                red_button_index = 1
                red_circle_index = 0

            red_button = (
                    int(circles[:, red_button_index, 0]),
                    int(circles[:, red_button_index, 1]),
                    int(circles[:, red_button_index, 2])
                    )
            red_circle = (
                    int(circles[:, red_circle_index, 0]),
                    int(circles[:, red_circle_index, 1]),
                    int(circles[:, red_circle_index, 2])
                    )
            cv2.circle(
                    self._img,
                    red_circle[0:2],
                    red_circle[2],
                     (0, 255, 0),
                     2)
            cv2.circle(
                    self._img,
                    red_button[0:2],
                    red_button[2],
                     (0, 255, 255),
                     2)
            cv2.circle(self._img, red_circle[0:2], 2, (0, 0, 255), 3)
            cv2.circle(self._img, red_button[0:2], 2, (0, 255, 255), 3)

            results = {
                    'red_button':  red_button[0:2],
                    'red_circle':  red_circle[0:2]
                    }
            return results

