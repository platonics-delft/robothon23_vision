import numpy as np
import cv2
from robothon23vision.utils.compute_tf import compute_transform

MIN_MATCH_COUNT = 2


class Localizer(object):

    def __init__(self, template):
        assert isinstance(template, str)
        self._full_template = cv2.imread(template, 0)


        self._pixel_m_factor = 1/0.0011559324339032173
        self._pixel_m_factor = 492/0.25
        self._pixel_m_factor = 700/0.25
        self._pixel_m_factor = 1/0.0005349688581191003


        h, w = self._full_template.shape
        cropped_h = [100, 600]
        cropped_w = [300, 980]
        self._template = self._full_template[
            cropped_h[0] : cropped_h[1], cropped_w[0] : cropped_w[1]
        ]
        self.delta_translation = np.float32([
            cropped_w[0],
            cropped_h[0],
        ])

    def set_image(self, img) -> None:
        self._img = img

    def detect_points(self):
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        # initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self._template, None)
        kp2, des2 = sift.detectAndCompute(gray, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # find matches by knn which calculates point distance in 128 dim
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good.append(m)

        # translate keypoints back to full source template
        for k in kp1:
            k.pt = (k.pt[0] + self.delta_translation[0], k.pt[1] + self.delta_translation[1])


        if len(good) > MIN_MATCH_COUNT:
            self._src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            self._dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        try:
            M, mask = cv2.findHomography(self._src_pts, self._dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matchesMask,
                flags=2,
            )
            self._annoted_image = cv2.drawMatches(self._full_template, kp1, self._img, kp2, good, None, **draw_params)
        except Exception as e:
            print(e)

    def annoted_image(self):
        return self._annoted_image

    def pixel_locations(self) -> dict:
        return self._dst_pts

    def compute_tf(self) -> np.ndarray:
        p0 = np.transpose(np.array(self._src_pts))[:, 0, :] - np.array(self._full_template.shape)[:, None]/2
        p1 = np.transpose(np.array(self._dst_pts))[:, 0, :] - np.array(self._full_template.shape)[:, None]/2
        T0 = compute_transform(p0, p1)
        #T0[0:2, 2] /= self._pixel_m_factor
        return T0

    def compute_full_tf_in_m(self) -> np.ndarray:
        T0 = self.compute_tf()
        T0[0:2, 2] /= self._pixel_m_factor
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


