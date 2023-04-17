# 导入工具包
import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def angle_between_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    return np.abs(angle1 - angle2) * 180 / np.pi


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return int(px), int(py)

def sort_vertices(vertices):
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]

def get_barcode(img):
    barcodes = pyzbar.decode(img)
    barcode = barcodes[0]
    barcode_data = barcode.data.decode("utf-8")
    return barcode_data

image = cv2.imread('example/5.jpg')
#坐标也会相同变化
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height = 500)
# 提高对比度
# image = cv2.addWeighted(image,1.5, image, 0 , 1.5)
# 预处理
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# thresh,gray = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# adaptive_threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 21, 20)
# 中值滤波
gray = cv2.medianBlur(gray, 13)
# 高斯滤波
gray = cv2.GaussianBlur(gray, (13, 13), 0)
# 使用Canny算子
edged = cv2.Canny(gray, 10, 100)
# 使用laplacian算子
# edged = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
# kernel = np.ones((19, 19), np.uint8)
# edged = cv2.dilate(edged, kernel, iterations=1)
# 作开运算
# edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# edged = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Gray", gray)
cv2.imshow("Edged", edged)
cv2.imwrite('example/Desk1_edged.jpg', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# # 遍历轮廓
# for c in cnts:
#     # 计算轮廓近似
#     peri = cv2.arcLength(c, True)
#     # c表示输入的点集
#     # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
#     # True表示封闭的
#     approx = cv2.approxPolyDP(c, 0.02*peri, False)
#     # 4个点的时候就拿出来
#     if len(approx) == 4:
#         screenCnt = approx
#         break
# 展示结果
# print("STEP 2: 获取轮廓")
# cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)
# # 绘制角点
# cv2.circle(image, tuple(screenCnt[0][0]), 5, (0, 0, 255), -1)
# cv2.circle(image, tuple(screenCnt[1][0]), 5, (0, 0, 255), -1)
# cv2.circle(image, tuple(screenCnt[2][0]), 5, (0, 0, 255), -1)
# cv2.circle(image, tuple(screenCnt[3][0]), 5, (0, 0, 255), -1)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 90, minLineLength=10, maxLineGap=10)

if lines is not None:
    selected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        selected_lines.append(line[0])

    # 按照角度进行排序
    sorted_lines = sorted(selected_lines, key=lambda l: np.arctan2(l[3] - l[1], l[2] - l[0]))

    # 选择四条边
    final_lines = [sorted_lines[0], sorted_lines[-1]]
    for i in range(1, len(sorted_lines) - 1):
        if angle_between_lines(sorted_lines[i], final_lines[-1]) > 0:
            final_lines.append(sorted_lines[i])
            if len(final_lines) == 4:
                break
    
    
    # 获取四个顶点
    vertices = []
    for i in range(len(final_lines)):
        for j in range(len(final_lines)):
            intersection = line_intersection(final_lines[i], final_lines[j])
            if i!=j:
                vertices.append(intersection)
    
    # 绘制点
    for vertex in vertices:
        cv2.circle(image, vertex, 5, (0, 0, 255), -1)
        print(vertex)
    print(len(vertices))
    
    #删除重复的点和坐标x或y为负数的点
    vertices = list(set(vertices))
    print(len(vertices))
    # 删除x或y小于0的顶点
    filtered_vertices = [vertex for vertex in vertices if vertex[0] >= 0 and vertex[1] >= 0 and vertex[0] <= image.shape[1] and vertex[1] <= image.shape[0]]

    print(len(filtered_vertices))
    print(filtered_vertices)
    # 绘制四边形
    if len(filtered_vertices) == 4:
        # 对顶点进行排序
        sorted_vertices = sort_vertices(np.array(filtered_vertices))

        # 绘制四边形
        cv2.polylines(image, [sorted_vertices.astype(np.int32)], True, (0, 0, 255), 2)
# print(get_barcode(image))
if len(filtered_vertices) == 4:
    # 对顶点进行排序
    sorted_vertices = sort_vertices(np.array(filtered_vertices))
    # 获取变换矩阵
    M = cv2.getPerspectiveTransform(sorted_vertices.astype(np.float32), np.array([(0, 0), (500, 0), (500, 500), (0, 500)]).astype(np.float32))
    # 进行仿射变换
    # warped = cv2.warpPerspective(image, M, (500, 500))
    # 输出矩形图片
    # warped = cv2.addWeighted(warped,1.5, warped, 0 , 1.5)
    # cv2.imshow("Warped", warped)
    # print(get_barcode(warped))
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()