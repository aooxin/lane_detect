import cv2
import numpy as np


def calculate_slope(line):
    """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    # print((y_2 - y_1) / (x_2 - x_1))
    return (y_2 - y_1) / (x_2 - x_1)


edge_img = cv2.imread("masked_edge_img.jpg", cv2.IMREAD_GRAYSCALE)
# 获取所有线段
lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=35, maxLineGap=20)
# 按照斜率分成车道线
left_lines = [line for line in lines if calculate_slope(line) > 0]
i = -1
for line in left_lines:
    i = i + 1
    if calculate_slope(line) <= 0.26:
        left_lines.pop(i)
right_lines = [line for line in lines if calculate_slope(line) < 0]
for line in right_lines:
    print(calculate_slope(line))
print(len(right_lines))
i = -1
for line in right_lines:
    i = i + 1
    if calculate_slope(line) >= -0.26:
        right_lines.pop(i)
for line in right_lines:
    print(calculate_slope(line))
print(len(right_lines))
