import cv2

# 读灰度图
img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("img_gray", img)

if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows()
    cv2.imwrite("img_gray.bmp", img)
