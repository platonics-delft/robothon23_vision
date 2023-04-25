import cv2
import numpy as np
import time
np.set_printoptions(suppress=True)

def match(template, img):
    img_g = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_g = template#cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template_g, None)
    kp2, des2 = sift.detectAndCompute(img_g, None)

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

    if len(good) > 2:
        _src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        _dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    try:
        M, mask = cv2.findHomography(_src_pts, _dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=2,
        )

        test = cv2.estimateAffinePartial2D(
            _src_pts,
            _dst_pts,
            method=cv2.RANSAC,
            maxIters=3,
            confidence=0.99,
            refineIters=10
        )
        annotated_image = cv2.drawMatches(template, kp1, img, kp2, good, None, **draw_params)
    except Exception as e:
        print(e)

    return annotated_image

if __name__ == '__main__':
    with np.load('probe_placing.npz') as data:
        print(data.files)
        imgs = data['img']

    t0 = time.time()
    times = 0 
    for i in range(0, imgs.shape[0]-100, 20):
        annotated_img = match(imgs[i, :, :], imgs[i+10, :, :])
        times+=1


        cv2.imshow('test', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"{times/((time.time()-t0))}")

