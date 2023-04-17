import cv2
import numpy as np
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



# 读取图像
image = cv2.imread('example/5.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height = 500)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


cv2.imshow('hsv',hsv_image)
# cv2.setMouseCallbxck("imageHSV", getpos)
cv2.waitKey(0)
a = hsv_image.shape[0]
b = hsv_image.shape[1]
h = []
s = []
v = []
# 定义快递单颜色范围（例如，白色快递单）
# lower_color = np.array([160, 5, 50])  # 根据具体情况调整
# upper_color = np.array([200, 20, 100])  # 根据具体情况调整
for i in range(200,409):
    for j in range(113,200):
        h.append(hsv_image[i, j, 0])
        s.append(hsv_image[i, j, 1])
        v.append(hsv_image[i, j, 2])
H = np.array(h)
S = np.array(s)
V = np.array(v)
h_min = np.min(H)
h_max = np.max(H)
s_min = np.min(S)
s_max = np.max(S)
v_min = np.min(V)
v_max = np.max(V)
print(h_min,h_max,s_min,s_max,v_min,v_max)
lower_color = np.array([h_min, s_min, v_min])  # 根据具体情况调整
upper_color = np.array([h_max, s_max, v_max])  # 根据具体情况调整

# 创建颜色掩码
mask = cv2.inRange(hsv_image, lower_color, upper_color)
# 高斯滤波
mask = cv2.GaussianBlur(mask, (13, 13), 0)
# 对掩码进行形态学操作
kernel = np.ones((15, 10), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations=1)
erosion = cv2.erode(dilation, kernel, iterations=1)

# Canny边缘检测
edges = cv2.Canny(erosion, 150, 350)
cv2.imshow('hsv_image', edges)
cv2.waitKey(0)

# # 保存结果
# cv2.imwrite('output_image.jpg', image)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 65, minLineLength=65, maxLineGap=100)
for line in lines:
    cv2.line(image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 2)
cv2.imshow('hsv_image', image) 
if lines is not None:
    selected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if(length > 40):
            selected_lines.append(line[0])

    # 按照角度进行排序
    sorted_lines = sorted(selected_lines, key=lambda l: np.arctan2(l[3] - l[1], l[2] - l[0]))
    if sorted_lines is not None:
        for line in sorted_lines:
            cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
        cv2.imshow('hsv_image', image) 
    #将角度小于10度的线段合并
    
    
    # 选择四条
    final_lines = [sorted_lines[0], sorted_lines[-1]]
    for i in range(1, len(sorted_lines) - 1):
        if angle_between_lines(sorted_lines[i], final_lines[-1]) > 10:
            final_lines.append(sorted_lines[i])
            if len(final_lines) == 5:
                break
        else :
            print('final_lines is None')
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

# cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# # 遍历轮廓
# for c in cnts:
#     # 计算轮廓近似
#     peri = cv2.arcLength(c, True)
#     # c表示输入的点集
#     # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
#     # True表示封闭的
#     approx = cv2.approxPolyDP(c, 0.02, True)
#     # 4个点的时候就拿出来
#     if len(approx) == 4:
#         screenCnt = approx
#         break
# # 展示结果
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