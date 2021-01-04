import cv2
import numpy as np
import math
from scipy import misc, ndimage

def jiaozheng(img, threshold2 = 170):
    """
    图像微调
    :param img:
    :return:
    """
    gray = cv2.cvtColor(img[10:-10,::], cv2.COLOR_BGR2GRAY)
    #li 150 -> 170
    # 50 -> 40
    edges = cv2.Canny(gray, 50, threshold2, apertureSize=3)

    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        # print('theta=',theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        pass
    t = float(y2 - y1) / max((x2 - x1),1)
    rotate_angle = math.degrees(math.atan(t))
    # print('t=', t)
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle,cval=255)
    return rotate_img, rotate_angle


def rotate_bound(image, angle):
    """
    图像旋转
    :param image:
    :param angle:
    :return:
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
