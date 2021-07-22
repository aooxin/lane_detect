import cv2


def get_edge_img(
    color_img,
    gaussian_ksize=5,
    gaussian_sigmax=1,
    canny_threshold1=50,
    canny_threshold2=100,
):

    """
    灰度化,模糊,canny变换,提取边缘
    :param color_img: 彩色图,channels=3
    """
    gaussian = cv2.GaussianBlur(
        color_img, (2 * gaussian_ksize - 1, gaussian_ksize), gaussian_sigmax
    )
    gray_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)
    return edges_img


img = cv2.imread("img.jpg", cv2.IMREAD_COLOR)

edge_img = get_edge_img(img)

cv2.imwrite("edges_img.jpg", edge_img)
cv2.imshow("edges", edge_img)
cv2.waitKey(0)
