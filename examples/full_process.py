import sys
import numpy as np
import cv2
from robothon23vision.localization.localizer import Localizer


def main():
    file_name_0 = sys.argv[1]
    img_0 = cv2.imread(file_name_0)
    localizer_0 = Localizer(img_0)
    localizer_0.detect_points()

    """
    cv2.imshow('red_image', localizer_0.red_image())
    cv2.waitKey(0)
    cv2.imshow('blue_image', localizer_0.blue_image())
    cv2.waitKey(0)
    cv2.imshow('red_image_0', localizer_1.red_image())
    cv2.waitKey(0)
    cv2.imshow('blue_image_0', localizer_1.blue_image())
    cv2.waitKey(0)
    """


    print(localizer_0.points())
    T0 = localizer_0.compute_tf()
    T_gio = localizer_0.compute_full_tf_in_m()
    print(T_gio)
    p0 = localizer_0.ground_truth_points()
    n = p0.shape[1]
    p0_aug = np.concatenate((p0, np.ones((1, n))), axis=0)
    p0_for = np.dot(T0, p0_aug)
    for i in range(p0_for.shape[1]):
        # Draw the outer circle
        center = (int(p0_for[0, i]), int(p0_for[1, i]))
        cv2.circle(img_0, center, 10, (0, 0, 255), 2)
        cv2.circle(img_0, center, 2, (0, 0, 255), 3)
    p1 = localizer_0.points()
    for i in range(p1.shape[1]):
        center = (int(p1[0, i]), int(p1[1, i]))
        cv2.circle(img_0, center, 10, (0, 255, 255), 2)
        cv2.circle(img_0, center, 2, (0, 255, 255), 3)

    cv2.imshow('Thresholded Image', img_0)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

