# 导入工具包
import numpy as np
import cv2
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


image = cv2.imread('example/Desk1.jpg')
#坐标也会相同变化
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height = 500)
# 提高对比度
#image = cv2.addWeighted(image, 1, image, -0.2 , 1)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#adaptive_threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 21, 20)
gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
dilate_img = cv2.dilate(gray, kernel)
erode_img = cv2.erode(gray, kernel) 

"""
我选了一张较好的图片，有的图片要去噪（高斯模糊）
将两幅图像相减获得边；cv2.absdiff参数：(膨胀后的图像，腐蚀后的图像)
上面得到的结果是灰度图，将其二值化以便观察结果
反色，对二值图每个像素取反
"""
absdiff_img = cv2.absdiff(dilate_img,erode_img);
retval, threshold_img = cv2.threshold(absdiff_img, 15, 255, cv2.THRESH_BINARY); 
result = cv2.bitwise_not(threshold_img); 

cv2.imshow("jianzhu",image)
cv2.imshow("dilate_img",dilate_img)
cv2.imshow("erode_img",erode_img)
cv2.imshow("absdiff_img",absdiff_img)
cv2.imshow("threshold_img",threshold_img)
cv2.imshow("result",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
