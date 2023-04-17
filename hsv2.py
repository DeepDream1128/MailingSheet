import numpy as np
import cv2 as cv
# 设置putText函数字体
font=cv.FONT_HERSHEY_SIMPLEX
#计算两边夹角额cos值
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    a = hsv_image.shape[0]
    b = hsv_image.shape[1]
    h = []
    s = []
    v = []
    # 定义快递单颜色范围（例如，白色快递单）
    # lower_color = np.array([160, 5, 50])  # 根据具体情况调整
    # upper_color = np.array([200, 20, 100])  # 根据具体情况调整
    for i in range(109,428):
        for j in range(73,299):
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
    mask = cv.inRange(hsv_image, lower_color, upper_color)
    # 高斯滤波
    mask = cv.GaussianBlur(mask, (13, 13), 0)
    # 对掩码进行形态学操作
    kernel = np.ones((15, 10), np.uint8)
    dilation = cv.dilate(mask, kernel, iterations=1)
    erosion = cv.erode(dilation, kernel, iterations=1)

    # Canny边缘检测
    edges = cv.Canny(erosion, 150, 350)
    squares = []
    # img = cv.GaussianBlur(img, (3, 3), 0)   
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # bin = cv.Canny(gray, 30, 100, apertureSize=3)    
    contours, _hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("轮廓数量：%d" % len(contours))
    index = 0
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True) #计算轮廓周长
        cnt = cv.approxPolyDP(cnt, 0.07*cnt_len, True) #多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        print(len(cnt))
        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            M = cv.moments(cnt) #计算轮廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#轮廓重心
            
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            #if max_cos < 0.1:
            # 检测四边形（不限定角度范围）
            if True:
                index = index + 1
                cv.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
            print(squares)
    return squares, img

def resize(image, width=None, height=None, inter=cv.INTER_AREA):
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
    resized = cv.resize(image, dim, interpolation=inter)
    return resized

def main():
    image = cv.imread('example/Desk1.jpg')
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = resize(orig, height = 500)
    # hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # 查找轮廓
    squares,image=find_squares(image)
    cv.drawContours(image, squares, -1, (0, 0, 255), 2 )
    cv.imshow('squares',image)
    cv.waitKey(0)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()