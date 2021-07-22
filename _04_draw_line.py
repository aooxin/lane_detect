import cv2
import numpy as np

img = np.zeros((200, 400))
cv2.line(img, (10, 10), (200, 100), 255, 3)
cv2.line(img, (30, 50), (350, 10), 255, 2)

cv2.imshow('img', img)
cv2.imwrite('lines.jpg', img)
cv2.waitKey(0)


