import cv2
import numpy as np

pre_lines = ()


def get_edge_img(
    color_img,
    gaussian_ksize=5,
    gaussian_sigmax=1,
    canny_threshold1=50,
    canny_threshold2=100,
):

    """
    灰度化,模糊,canny变换,提取边缘
    """
    gaussian = cv2.GaussianBlur(
        color_img, (gaussian_ksize, gaussian_ksize), gaussian_sigmax
    )
    gray_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)
    return edges_img


def test_guss(
    color_img,
    gaussian_ksize=5,
    gaussian_sigmax=1,
):
    gaussian = cv2.GaussianBlur(
        color_img, (gaussian_ksize, gaussian_ksize), gaussian_sigmax
    )
    return gaussian


def get_edge_img_2(
    bin_img,
    canny_threshold1=50,
    canny_threshold2=100,
):
    edges_img = cv2.Canny(bin_img, canny_threshold1, canny_threshold2)
    return edges_img


def get_bin_img_1(color_img):
    """
    局部自适应阈值二值化
    """
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )
    return binary


def get_bin_img_2(image):
    """
    全局自适应阈值二值化
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE
    )  # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    return binary


def roi_mask(gray_img):
    """
    对gray_img进行掩膜
    :param gray_img: 灰度图,channels=1
    """
    # poly_pts = np.array([[[125, 324], [235, 259], [325, 259], [435, 324]]]) #视频1
    poly_pts = np.array([[[118, 243], [293, 112], [365, 110], [575, 243]]])  # 视频2
    mask = np.zeros_like(gray_img)
    mask = cv2.fillPoly(mask, pts=poly_pts, color=255)
    img_mask = cv2.bitwise_and(gray_img, mask)
    return img_mask


def calculate_slope(line):
    """
    计算线段line的斜率
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)


def get_lines(edge_img):
    def reject_abnormal_lines(lines, threshold=0.2):
        """
        剔除不一致的线段
        """
        slopes = [calculate_slope(line) for line in lines]
        while len(lines) > 0:
            mean = np.mean(slopes)
            diff = [abs(s - mean) for s in slopes]
            idx = np.argmax(diff)
            if diff[idx] > threshold:
                slopes.pop(idx)
                lines.pop(idx)
            else:
                break
        return lines

    def least_squares_fit(lines):
        """
        线段拟合
        """
        try:
            x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
            y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
            poly = np.polyfit(x_coords, y_coords, deg=1)
            point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
            point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
            return np.array([point_min, point_max], dtype=np.int)
        except BaseException:
            # print("UnFound")
            return

    # 获取所有线段
    try:
        lines = cv2.HoughLinesP(
            edge_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20
        )
        # 按照斜率分成车道线
        left_lines = [line for line in lines if calculate_slope(line) > 0]
        i = -1
        for line in left_lines:
            i = i + 1
            if calculate_slope(line) <= 0.25:
                left_lines.pop(i)
        right_lines = [line for line in lines if calculate_slope(line) < 0]
        i = -1
        for line in right_lines:
            i = i + 1
            if calculate_slope(line) >= -0.25:
                right_lines.pop(i)
        # 剔除离群线段
        left_lines = reject_abnormal_lines(left_lines)
        right_lines = reject_abnormal_lines(right_lines)

        return least_squares_fit(left_lines), least_squares_fit(right_lines)
    except BaseException:
        # print("get_error")
        return


def draw_lines(img, lines):
    """
    绘制线段
    """
    try:
        x = False
        global pre_lines
        left_line, right_line = lines
        # print(left_line)
        # print((left_line[0][1] - left_line[1][1]) / (left_line[0][0] - left_line[1][0]))
        if (
            (left_line[0][1] - left_line[1][1]) / (left_line[0][0] - left_line[1][0])
        ) > 0.5:
            """print(
                (left_line[0][1] - left_line[1][1])
                / (left_line[0][0] - left_line[1][0])
            )"""
            cv2.line(
                img,
                tuple(left_line[0]),
                tuple(left_line[1]),
                color=(0, 255, 255),
                thickness=4,
            )
            x = True
        else:
            left_line, right_line = pre_lines
            if (
                (left_line[0][1] - left_line[1][1])
                / (left_line[0][0] - left_line[1][0])
            ) > 0.5:
                cv2.line(
                    img,
                    tuple(left_line[0]),
                    tuple(left_line[1]),
                    color=(0, 255, 255),
                    thickness=4,
                )
                x = False
        # print(right_line)
        if (
            (right_line[0][1] - right_line[1][1])
            / (right_line[0][0] - right_line[1][0])
        ) < -0.5:
            cv2.line(
                img,
                tuple(right_line[0]),
                tuple(right_line[1]),
                color=(0, 255, 255),
                thickness=4,
            )
            x = True
        else:
            left_line, right_line = pre_lines
            if (
                (right_line[0][1] - right_line[1][1])
                / (right_line[0][0] - right_line[1][0])
            ) < -0.5:
                cv2.line(
                    img,
                    tuple(right_line[0]),
                    tuple(right_line[1]),
                    color=(0, 255, 255),
                    thickness=4,
                )
                x = False
        if x == True:
            pre_lines = lines
    except BaseException:
        return


def do_do_do(color_img):
    # global pre_lines
    edge_img = get_edge_img(color_img)
    mask_img_gray = roi_mask(edge_img)
    lines = get_lines(mask_img_gray)
    draw_lines(color_img, lines)
    return color_img
    return mask_img_gray
    return edge_img


def test(color_img):
    binary_img = get_bin_img_1(color_img)
    edge_img = get_edge_img_2(binary_img)
    mask_img_gray = roi_mask(edge_img)
    lines = get_lines(mask_img_gray)
    draw_lines(color_img, lines)
    # return edge_img
    return color_img
    return binary_img


def test_b_e(color_img):
    """先边缘再二值化"""
    img = do_do_do(color_img)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )
    mask_img_gray = roi_mask(binary)
    lines = get_lines(mask_img_gray)
    draw_lines(color_img, lines)
    # return mask_img_gray
    # return edge_img
    return color_img
    return binary


# cap = cv2.VideoCapture("../video2.mp4")
cap = cv2.VideoCapture("../video3.mp4")
while True:
    ret, frame = cap.read()
    frame = do_do_do(frame)  # 不进行二值化处理
    # print(frame.shape)
    # frame = test(frame) # 先进行了二值化处理
    # frame = test_guss(frame) # 测试高斯
    # frame = test_b_e(frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(10)
