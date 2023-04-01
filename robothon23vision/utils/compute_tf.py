import numpy as np
from scipy.optimize import least_squares
import numpy as np

def compute_transform(points: np.ndarray, transformed_points: np.ndarray):
    assert isinstance(points, np.ndarray)
    assert isinstance(transformed_points, np.ndarray)
    assert points.shape == transformed_points.shape
    n = int(points.shape[1])
    points_out = np.concatenate((transformed_points, np.ones((1, n))), axis=0)
    points_in = np.concatenate((points, np.ones((1, n))), axis=0)
    return np.dot(points_out, np.linalg.pinv(points_in))

def main():

    theta = 0.3
    x = 100
    y = 0
    c = np.cos(theta)
    s = np.sin(theta)
    T_true = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

    points = np.transpose(np.array([
        [340, 30],
        [240, 20],
        [143, 73],
        [47, 202],
        [840, 152],
        [740, 20],
        [37, 202],
        [341, 152],
        [240, 17],
        [179, 117],
        ]))

    noise = np.random.random(points.shape) * 10

    n = int(points.shape[1])
    points_augmented = np.concatenate((points, np.ones((1, n))), axis=0)
    points_tf = np.dot(T_true, points_augmented)[:2, :] + noise

    #T_star = np.transpose(compute_transform(np.transpose(points), np.transpose(points_tf)))
    T_star = compute_transform(points, points_tf)


    print(T_true)
    print(T_star)
    error = np.linalg.norm(T_true-T_star)
    print(error)

if __name__ == "__main__":
    main()
