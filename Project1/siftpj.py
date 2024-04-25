import cv2 as cv
import numpy as np


# 检测图像的SIFT关键特征点
def siftjiance(image):
    # 将图像转换为灰度图
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 获取图像特征sift-SIFT特征点,实例化对象sift  
    sift = cv.SIFT_create()

    keypoints, features = sift.detectAndCompute(image, None)

    keypoints_image = cv.drawKeypoints(
        gray_image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    return keypoints_image, keypoints, features


# 使用KNN检测来自左右图像的SIFT特征，随后进行匹配
def pointmatch(features_right, features_left):
    # 创建BFMatcher对象解决匹配  
    bf = cv.BFMatcher()

    matches = bf.knnMatch(features_right, features_left, k=2)
    # 利用sorted()函数对matches对象进行升序(默认)操作  
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)

    # 建立列表dots用于存储匹配的点集
    dots = []
    for m, n in matches:
        # ratio的值越大，匹配的线条越密集，但错误匹配点也会增多
        ratio = 0.6
        if m.distance < ratio * n.distance:
            dots.append(m)
            # 返回匹配的关键特征点集
    return dots


# 计算视角变换矩阵H，用H对右图进行变换并返回全景拼接图像
def pinjie(image_right, image_left):
    _, keypoints_right, features_right = siftjiance(image_right)
    _, keypoints_left, features_left = siftjiance(image_left)
    goodMatch = pointmatch(features_right, features_left)
    # 当筛选项的匹配对大于4对(因为homography单应性矩阵的计算需要至少四个点)时,计算视角变换矩阵  
    if len(goodMatch) > 4:
        # 获取匹配对的点坐标  
        ptsR = np.float32(
            [keypoints_right[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsL = np.float32(
            [keypoints_left[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        # ransacReprojThreshold：将点对视为内点的最大允许重投影错误阈值(仅用于RANSAC和RHO方法时),若srcPoints和dstPoints是以像素为单位的，该参数通常设置在1到10的范围内  
        ransacReprojThreshold = 4
        # cv.findHomography():计算多个二维点对之间的最优单映射变换矩阵 H(3行x3列),使用最小均方误差或者RANSAC方法  
        # 函数作用:利用基于RANSAC的鲁棒算法选择最优的四组配对点，再计算转换矩阵H(3*3)并返回,以便于反向投影错误率达到最小  
        Homography, status = cv.findHomography(
            ptsR, ptsL, cv.RANSAC, ransacReprojThreshold)
        # cv.warpPerspective()：透视变换函数，用于解决cv2.warpAffine()不能处理视场和图像不平行的问题  

        jieguo = cv.warpPerspective(
            image_right, Homography, (image_right.shape[1] + image_left.shape[1], image_right.shape[0]))

        # 将左图加入到变换后的右图像的左端即获得最终图像  
        jieguo[0:image_left.shape[0], 0:image_left.shape[1]] = image_left
        # 返回全景拼接的图像 
        return jieguo


if __name__ == '__main__':

    image_left = cv.imread("5.png")
    image_right = cv.imread("6.png")
    # 通过调用cv2.resize()使用插值的方式来改变图像的尺寸，保证左右两张图像大小一致  

    image_right = cv.resize(image_right, None, fx=1, fy=1)
    image_left = cv.resize(image_left, (image_right.shape[1], image_right.shape[0]))
    # 获取检测到关键特征点后的图像的相关参数  
    keypoints_image_right, keypoints_right, features_right = siftjiance(image_right)
    keypoints_image_left, keypoints_left, features_left = siftjiance(image_left)
    # 利用np.hstack()函数同时将原图和绘有关键特征点的图像沿着竖直方向(水平顺序)堆叠起来  
    cv.imshow("leftpoint", np.hstack((image_left, keypoints_image_left)))

    cv.waitKey(0)

    cv.destroyAllWindows()
    cv.imshow("rightpoint", np.hstack((image_right, keypoints_image_right)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    goodMatch = pointmatch(features_right, features_left)


    match_image = cv.drawMatches(
        image_right, keypoints_right, image_left, keypoints_left, goodMatch, None, None, None, None, flags=2)
    cv.imshow("match", match_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 把图片拼接成全景图
    jieguo = pinjie(image_right, image_left)

    cv.imshow("result", jieguo)

    cv.waitKey(0)
    cv.destroyAllWindows()
