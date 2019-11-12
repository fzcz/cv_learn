import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 4

img1 = cv2.imread("C://CV/image/room1.jpg", 0)
img2 = cv2.imread("C://CV/image/room2.jpg", 0)

#使用SIFT检测角点
sift = cv2.xfeatures2d.SIFT_create()
#获取关键点和描述
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#定义FLANN匹配器
index_params = dict(algorithm = 1, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

#使用knn算法匹配
good = []
matches = flann.knnMatch(des1, des2, k = 2)

#去除错误匹配
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)


#单应性
if len(good)> MIN_MATCH_COUNT:
    #改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #findHomography函数是计算变换矩阵
    #参数cv2,RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H, 即返回值M
    #返回值：M 为变换矩阵， mask 是掩模
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)
    #ravel方法将数据降维处理，最后转换成列表格式
    matchesMask = mask.ravel().tolist()
    #获取img1图像的尺寸
    h, w = img1.shape
    #获取img2图像的尺寸
    h1, w1 = img1.shape
    #pts是图像img1的四个顶点，pts2是图像img2的四个顶点
    pts1 = np.float32([[0,0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0,0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)

    #计算变换后的四个顶点坐标,
    dst = cv2.perspectiveTransform(pts1, M)

    #实现img1和img2两个图像的简单拼接
    total_dst = np.concatenate((pts2, dst), axis=0)
    min_x,min_y = np.int32(total_dst.min(axis=0).ravel())
    max_x, max_y = np.int32(total_dst.max(axis=0).ravel())
    shift_to_zero_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    trans_img1 = cv2.warpPerspective(img1, shift_to_zero_matrix.dot(M), (max_x - min_x, max_y - min_y))
    trans_img1[-min_y:h1 - min_y, -min_x:w1 - min_x] = img2
    img = cv2.resize(trans_img1, (666,666))
    cv2.imshow("stitch", img)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
    #根据四个顶点坐标的位置在img2上画出变换后的边框
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (255,0, 0), 3, cv2.LINE_AA)
else:
    print(" Not enough matches are found - %d%d" %(len(good), MIN_MATCH_COUNT))
    matchesMask = None


#显示匹配结果
draw_params = dict(matchColor = (0, 255, 0), #用绿线连接匹配点
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3  = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, "gray"), plt.show()