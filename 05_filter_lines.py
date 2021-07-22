import cv2
import numpy as np


def calculate_slope(line):
    """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)


edge_img = cv2.imread("masked_edge_img.jpg", cv2.IMREAD_GRAYSCALE)
# 获取所有线段
lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=6, maxLineGap=60)
# 按照斜率分成车道线
left_lines = [line for line in lines if calculate_slope(line) > 0]
right_lines = [line for line in lines if calculate_slope(line) < 0]


def reject_abnormal_lines(lines, threshold):
    """
    剔除斜率不一致的线段
    :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    """
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)
        # 平均
        diff = [abs(s - mean) for s in slopes]
        # 差值
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines


print("before filter:")
print("left lines number=")
print(len(left_lines))
print("right lines number=")
print(len(right_lines))

reject_abnormal_lines(left_lines, threshold=0.2)
reject_abnormal_lines(right_lines, threshold=0.2)


print("after filter:")
print("left lines number=")
print(len(left_lines))
print("right lines number=")
print(len(right_lines))
