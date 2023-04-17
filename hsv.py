import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('example/17_898.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义快递单颜色范围（例如，白色快递单）
lower_color = np.array([0, 0, 200])  # 根据具体情况调整
upper_color = np.array([180, 50, 255])  # 根据具体情况调整

# 创建颜色掩码
mask = cv2.inRange(hsv_image, lower_color, upper_color)

# 对掩码进行形态学操作
kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations=1)
erosion = cv2.erode(dilation, kernel, iterations=1)

# Canny边缘检测
edges = cv2.Canny(erosion, 100, 200)

# 将edges转为二值化图像

cv2.imshow('edges', edges)
cv2.waitKey(0)
# 霍夫变换检测直线
lines =cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength=100, maxLineGap=10)

# 在原图上绘制直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 使用Harris角点检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# 放大角点检测结果
corners = cv2.dilate(corners, None)

# 在原图上标记角点
image[corners > 0.01 * corners.max()] = [0, 255, 0]

# 显示结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# 保存结果
cv2.imwrite('output_image.jpg', image)
