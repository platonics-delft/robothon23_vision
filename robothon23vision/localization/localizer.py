import numpy as np
from robothon23vision.detectors.blue_area_detector import BlueAreaDetector
from robothon23vision.detectors.blue_button_detector import BlueButtonDetector
from robothon23vision.detectors.errors import DetectionError
from robothon23vision.detectors.gray_area_detector import GrayAreaDetector
from robothon23vision.detectors.red_button_detector import RedButtonDetector
from robothon23vision.utils.compute_tf import compute_transform

feature_map = {'gray_area': 0, 'red_button': 1, 'blue_button': 2, 'blue_area': 3}

class Localizer(object):

    def __init__(self, debug=False):
        self._detected_points = {}
        self._reference_points = {}
        self._debug = debug
        self._ground_truth = {
                'gray_area': (329, 168),
                'red_button': (521, 158),
                'blue_button': (522, 138),
                'blue_area': (523, 215),
        }
        self._pixel_cm_factor = 492.0/0.25
        self._distance_matrix = self.compute_distance_matrix(self._ground_truth)
        print(self._distance_matrix)
        self._points = {}

    def compute_distance_matrix(self, points: dict) -> np.ndarray:
        distance_matrix = np.ones((4, 4)) * 10000
        for key_i, item_i in points.items():
            for key_j, item_j in points.items():
                dist_ij = np.linalg.norm(np.array(item_i) - np.array(item_j))
                i = feature_map[key_i]
                j = feature_map[key_j]
                distance_matrix[i, j] = dist_ij
                distance_matrix[j, i] = dist_ij
        return distance_matrix

    def set_ground_truth(self, ground_truth_locations: dict) -> None:
        self._ground_truth = ground_truth_locations

    def set_image(self, img) -> None:
        self._img = img
        self._detectors = []
        self._detectors.append(RedButtonDetector(self._img))
        self._detectors.append(BlueButtonDetector(self._img))
        self._detectors.append(GrayAreaDetector(self._img))
        self._detectors.append(BlueAreaDetector(self._img))

        for detector in self._detectors:
            detector.set_mask()

    def detect_points(self):
        points = {}
        for detector in self._detectors:
            try:
                res = detector.detect()
                points.update(res)
            except DetectionError as e:
                if self._debug:
                    print(e)
        print(points)
        distance_matrix = self.compute_distance_matrix(points)
        difference_matrix = np.abs(distance_matrix - self._distance_matrix)
        index = difference_matrix < 100
        #print(difference_matrix)

        self._points = points

    def red_image(self):
        return self._red_img

    def blue_image(self):
        return self._blue_img

    def pixel_locations(self) -> dict:
        return self._detected_points

    def set_coordinate_locations(self, feature_locations: dict) -> None:
        self._feature_locations = feature_locations


    def points(self) -> np.ndarray:
        values = list(self._points.values())
        return np.transpose(np.array(values))

    def ground_truth_points(self) -> np.ndarray:
        values = list(self._ground_truth.values())
        return np.transpose(np.array(values))

    def compute_tf(self) -> np.ndarray:
        p0 = []
        p1 = []
        for key, value in self._points.items():
            p0.append(self._ground_truth[key])
            value_tf = [value[0] - 640, value[1] - 360]
            p1.append(value_tf)
        p0 = np.transpose(np.array(p0))
        p1 = np.transpose(np.array(p1))
        T0 = compute_transform(p0, p1)
        #T0[0:2, 2] /= self._pixel_cm_factor
        return T0

    def compute_full_tf_in_m(self) -> np.ndarray:
        T0 = self.compute_tf()
        T0[0:2, 2] /= self._pixel_cm_factor
        T_gio = np.identity(4)
        T_gio[0:2, 0:2] = T0[0:2, 0:2]
        T_gio[0:2, 3] = T0[0:2, 2]
        return T_gio

    def compute_coordinate_tf(self) -> np.ndarray:
        p0 = []
        p1 = []
        for key, value in self._feature_locations.items():
            p0.append(list(self._ground_truth_locations[key])[0:3])
            p1.append(list(value[0:3]))
        p0 = np.transpose(np.array(p0))
        p1 = np.transpose(np.array(p1))
        T0 = compute_transform(p0, p1)
        return T0


