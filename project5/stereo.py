import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取左右图像
img_left = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)


# 创建StereoBM对象
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


 # 计算视差图
disparity = stereo.compute(img_left, img_right)

# 归一化视差图以便显示
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# 使用matplotlib显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('left')
plt.imshow(img_left, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('right')
plt.imshow(img_right, cmap='gray')

plt.figure(figsize=(10, 5))
plt.subplot(1, 1, 1)
plt.title('Disparity Map')
plt.imshow(disparity_normalized, cmap='jet')
plt.colorbar()
plt.show()
